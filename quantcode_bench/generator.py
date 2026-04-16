"""
Strategy generator supporting single-shot and agentic modes.

Single-shot: Model must generate correct strategy on first attempt
Agentic: Model can iterate with feedback up to max_turns
"""

import asyncio
import concurrent.futures
import json
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI

# Thread pool for running blocking eval (backtest + judge) without blocking the event loop
_eval_executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _tokenizer = None

from .reward import backtest_reward_fn


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or fallback to ~4 chars/token estimate."""
    if not text:
        return 0
    if _tokenizer:
        return len(_tokenizer.encode(text))
    # Fallback: rough estimate ~4 chars per token
    return len(text) // 4


def detect_repetition_loop(code: str, threshold: int = 15) -> Tuple[bool, int, str]:
    """
    Detect if code has repetition loops (signs of model looping).
    
    Args:
        code: Generated code string
        threshold: Minimum number of repetitions to consider as loop
        
    Returns:
        Tuple of (is_looped, max_repeats, most_repeated_line)
    """
    if not code:
        return False, 0, ""
    
    lines = code.split('\n')
    line_counts = Counter(lines)
    
    # Find most repeated non-empty line
    for line, count in line_counts.most_common():
        if line.strip():  # Skip empty lines
            if count > threshold:
                return True, count, line[:80]
            return False, count, line[:80]
    
    return False, 0, ""

# ─── System prompts ──────────────────────────────────────────────────────────

_CODE_EXAMPLES = """
import backtrader as bt

class TradingStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )
    
    def __init__(self):
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)
        self.order = None
        
    def next(self):
        if self.order:
            return
        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
        else:
            if self.crossover < 0:
                self.order = self.sell()
                
    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None
"""

SYSTEM_PROMPT_EN = """You are an expert in creating trading strategies for Backtrader library (Python).

CRITICAL RULES - VIOLATION WILL CAUSE FAILURE:
1. Import ONLY: "import backtrader as bt" - NO pandas, numpy, scipy, yfinance, etc.
2. Use ONLY these indicators (DO NOT invent new ones):
   - bt.indicators.SMA / bt.indicators.SimpleMovingAverage
   - bt.indicators.EMA / bt.indicators.ExponentialMovingAverage
   - bt.indicators.RSI
   - bt.indicators.MACD
   - bt.indicators.BollingerBands
   - bt.indicators.Stochastic
   - bt.indicators.ATR
   - bt.indicators.CrossOver
3. Class name MUST be: TradingStrategy(bt.Strategy)
4. Must have: __init__(), next(), notify_order()
5. Use self.order to track orders
6. Check "if not self.position:" before opening positions
7. Use ONLY self.data (NOT self.data0, self.data1, etc.)
8. Available data fields: self.data.close, self.data.open, self.data.high, self.data.low, self.data.volume
9. DO NOT use: self.data.columns, self.data.index, dates, timestamps
10. Create SIMPLE strategies that WILL generate signals
11. Output ONLY executable Python code, NO text/explanations/markdown

COMMON ERRORS TO AVOID:
- bt.indicators.Sector - DOES NOT EXIST
- bt.indicators.Momentum - DOES NOT EXIST, use RSI instead
- bt.indicators.SMA(self.data) - WRONG, need: bt.indicators.SMA(self.data.close)
- bt.indicators.SMA([data1, data2]) - WRONG, indicators take ONE series
- self.data.columns - WRONG, no pandas attributes
- self.data.index - WRONG, no pandas attributes
- bt.Date(2023, 1, 1) - WRONG, no date filtering
- self.crossover = bt.indicators.CrossOver(self.sma, self.sma) - WRONG, same inputs
- if len(self) < 100: return - WRONG syntax
- Complex statistical functions - KEEP IT SIMPLE!

CORRECT INDICATOR USAGE:
self.sma = bt.indicators.SMA(self.data.close, period=20)
self.rsi = bt.indicators.RSI(self.data.close, period=14)
self.macd = bt.indicators.MACD(self.data.close)
self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2.0)
self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

EXAMPLE - SMA Crossover:
""" + _CODE_EXAMPLES + """
OUTPUT FORMAT:
- ONLY Python code, nothing else
- NO markdown (no ```python or ```)
- NO explanations before/after code
- NO comments outside the code
- If task is complex, SIMPLIFY it to use only available indicators"""


# ─── User-facing prompts ─────────────────────────────────────────────────────

_USER_PROMPT_SINGLE_EN = """
USER REQUEST:
{prompt}

YOUR TASK:
Generate Python code for a Backtrader strategy based on the description above.
Output ONLY Python code, no explanations, no markdown.
You have ONLY ONE attempt - make it count!"""


_USER_PROMPT_AGENTIC_EN = """
USER REQUEST:
{prompt}

YOUR TASK:
Generate Python code for a Backtrader strategy based on the description.
Output ONLY Python code, no explanations, no markdown."""


# ─── Feedback templates ──────────────────────────────────────────────────────

_FEEDBACK_TEMPLATES = {
    "en": {
        "success": "SUCCESS! Strategy generated {trades} trades in {turn} turns.",
        "compilation_error": """COMPILATION ERROR (Turn {turn}/{max_turns})

Error: {error}

Common fixes:
- Add 'import backtrader as bt'
- Ensure class inherits from bt.Strategy
- Include __init__(), next(), notify_order() methods
- Use only supported indicators""",
        "runtime_error": """RUNTIME ERROR (Turn {turn}/{max_turns})

Error: {error}

Common fixes:
- Check indicator parameters
- Use self.data.close, not self.data
- Verify logic in next() method""",
        "no_trades": """NO TRADES (Turn {turn}/{max_turns})

Strategy compiled and ran but generated zero trades.

Common fixes:
- Check entry conditions are not too strict
- Use 'if not self.position:' before buy
- Ensure indicators generate signals
- Simplify conditions""",
        "partial": "PARTIAL SUCCESS (Turn {turn}/{max_turns}): {trades} trades generated"
    },

}

SYSTEM_PROMPT = SYSTEM_PROMPT_EN


class StrategyGenerator:
    """
    Strategy generator with single-shot and agentic modes.
    
    Single-shot mode: max_turns=1, no feedback
    Agentic mode: max_turns>1, iterative refinement with feedback
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 1,
        timeout: int = 300,
        trajectories_dir: Optional[str] = None,
        verbose: Optional[bool] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        max_tokens: int = 64000,
    ):
        """
        Initialize the strategy generator.
        
        Args:
            api_key: API key for LLM service
            base_url: Base URL for API
            model: Model name
            max_turns: Maximum generation attempts (1 for single-shot)
            timeout: Request timeout in seconds
            trajectories_dir: Directory to save generation trajectories
            verbose: Enable detailed per-strategy logging (default: True if trajectories_dir is set)
            extra_body: Extra parameters for the API request (e.g. for sglang thinking control)
            max_tokens: Maximum tokens in generation response
        """
        self.api_key = api_key or os.environ.get("GENERATOR_API_KEY", "")
        self.base_url = base_url or os.environ.get("GENERATOR_BASE_URL", "")
        self.model = model or os.environ.get("GENERATOR_MODEL", "")
        self.max_turns = max_turns
        self.timeout = timeout
        self.trajectories_dir = trajectories_dir
        self.extra_body = extra_body
        self.max_tokens = max_tokens
        # Verbose logging enabled by default when saving trajectories
        self.verbose = verbose if verbose is not None else (trajectories_dir is not None)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout
        )
        
        if trajectories_dir:
            os.makedirs(trajectories_dir, exist_ok=True)
    
    def _write_trajectory(self, strategy_id: int, result: Dict[str, Any]) -> None:
        """Write trajectory to JSONL file."""
        if not self.trajectories_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        filepath = os.path.join(
            self.trajectories_dir,
            f"strategy_{strategy_id}_{timestamp}_{session_id}.jsonl"
        )
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Session start event
                f.write(json.dumps({
                    "event_type": "session_start",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "task_id": strategy_id,
                    "task_question": result.get("prompt", "")[:500],
                    "max_turns": self.max_turns,
                    "mode": "single_shot" if self.max_turns == 1 else "agentic"
                }, ensure_ascii=False) + "\n")
                
                # Turn events
                for turn_info in result.get("turns_history", []):
                    f.write(json.dumps({
                        "event_type": "turn_complete",
                        "timestamp": datetime.now().isoformat(),
                        "session_id": session_id,
                        "task_id": strategy_id,
                        "turn_number": turn_info.get("turn", 1),
                        "max_turns": self.max_turns,
                        "reward": float(turn_info.get("reward", 0)),
                        "metadata": turn_info.get("metadata", {}),
                        "generated_code": turn_info.get("code", ""),
                        "feedback": turn_info.get("feedback", ""),
                        "turn_time": turn_info.get("generation_time", 0)
                    }, ensure_ascii=False) + "\n")
                
                # Session complete event
                f.write(json.dumps({
                    "event_type": "session_complete",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "task_id": strategy_id,
                    "total_turns": result.get("total_turns", 1),
                    "final_reward": float(result.get("best_reward", 0)),
                    "success": result.get("generation_success", False),
                    "total_time": result.get("total_generation_time", 0)
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write trajectory for strategy {strategy_id}: {e}")
    
    def _clean_code(self, code: str) -> str:
        """Clean generated code from markdown and extra text."""
        if not code:
            return ""
        
        # Remove markdown blocks
        if "```python" in code:
            parts = code.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
        elif "```" in code:
            parts = code.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1 and part.strip():
                    code = part.strip()
                    break
        
        # Remove text before first import
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ")):
                code = "\n".join(lines[i:])
                break
        
        return code.strip()
    
    def _build_feedback(self, metadata: Dict[str, Any], turn: int) -> str:
        """Build feedback message for next iteration."""
        compilation_ok = metadata.get("compilation_success", False)
        backtest_ok = metadata.get("backtest_success", False)
        has_trades = metadata.get("has_trades", False)
        error = metadata.get("error_message", "")
        trades = metadata.get("total_trades", 0)
        
        tpl = _FEEDBACK_TEMPLATES["en"]
        fmt = {"turn": turn, "max_turns": self.max_turns, "trades": trades, "error": error}
        
        if compilation_ok and backtest_ok and has_trades:
            return tpl["success"].format(**fmt)
        if not compilation_ok:
            return tpl["compilation_error"].format(**fmt)
        if not backtest_ok:
            return tpl["runtime_error"].format(**fmt)
        if not has_trades:
            return tpl["no_trades"].format(**fmt)
        return tpl["partial"].format(**fmt)
    
    def _test_strategy(
        self, code: str, strategy_id: int, prompt: str,
        yf_symbol: str = "AAPL", timeframe: str = "1d",
    ) -> tuple:
        """Test strategy using reward function (synchronous)."""
        task_info = {
            "question": prompt,
            "strategy_id": strategy_id,
            "yf_symbol": yf_symbol,
            "timeframe": timeframe,
        }
        return backtest_reward_fn(task_info, code)

    async def _test_strategy_async(
        self, code: str, strategy_id: int, prompt: str,
        yf_symbol: str = "AAPL", timeframe: str = "1d",
    ) -> tuple:
        """Async wrapper: runs backtest + judge in thread pool without blocking event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _eval_executor,
            self._test_strategy,
            code, strategy_id, prompt, yf_symbol, timeframe,
        )
    
    async def _generate_single_async(
        self,
        prompt: str,
        strategy_id: int,
        yf_symbol: str = "AAPL",
        timeframe: str = "1d",
    ) -> Dict[str, Any]:
        """Single-shot async generation."""
        if self.verbose:
            print(f"[STRATEGY {strategy_id}] Starting single-shot generation ({yf_symbol}@{timeframe})", flush=True)
        start_time = time.time()
        
        full_prompt = SYSTEM_PROMPT_EN + _USER_PROMPT_SINGLE_EN.format(prompt=prompt)
        
        generated_code = ""
        finish_reason = None
        last_error = None
        
        api_error = False
        try:
            # Retry logic for API calls (skip retries for timeouts)
            for retry in range(6):
                try:
                    request_params = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": full_prompt}],
                        "temperature": 0.1,
                        "max_tokens": self.max_tokens,
                    }
                    if self.extra_body:
                        request_params["extra_body"] = self.extra_body

                    response = await self.async_client.chat.completions.create(**request_params)
                    
                    generated_code = response.choices[0].message.content or ""
                    finish_reason = response.choices[0].finish_reason
                    
                    # Extract token counts from response
                    reasoning_tokens = 0
                    completion_tokens = 0
                    if hasattr(response, 'usage') and response.usage:
                        completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                        # Some APIs return reasoning_tokens in completion_tokens_details
                        if hasattr(response.usage, 'completion_tokens_details'):
                            details = response.usage.completion_tokens_details
                            if details and hasattr(details, 'reasoning_tokens'):
                                reasoning_tokens = getattr(details, 'reasoning_tokens', 0) or 0
                    
                    # If no usage info, estimate with tiktoken
                    if completion_tokens == 0 and generated_code:
                        completion_tokens = count_tokens(generated_code)
                    
                    if self.verbose:
                        print(f"[STRATEGY {strategy_id}] LLM response: {completion_tokens} tokens (reasoning: {reasoning_tokens}), finish_reason={finish_reason}", flush=True)
                    break
                    
                except Exception as e:
                    last_error = str(e).lower()
                    if self.verbose:
                        print(f"[STRATEGY {strategy_id}] LLM API error (attempt {retry+1}/6): {str(e)[:200]}", flush=True)
                    
                    # Don't retry on timeouts - they won't recover
                    is_timeout = any(x in last_error for x in ["timeout", "timed out", "504", "gateway"])
                    if is_timeout:
                        if self.verbose:
                            print(f"[STRATEGY {strategy_id}] ⚠️ Timeout detected, skipping retries", flush=True)
                        api_error = True
                        generated_code = ""
                        break
                    
                    if retry < 5:
                        await asyncio.sleep(2)  # Reduced from 5s to 2s
                    else:
                        api_error = True
                        generated_code = ""
            
            if not generated_code:
                total_time = time.time() - start_time
                error_msg = "API timeout" if api_error else f"Empty response. finish_reason={finish_reason}"
                if self.verbose:
                    print(f"[STRATEGY {strategy_id}] {error_msg}, reward=0.0", flush=True)
                return {
                    "strategy_id": strategy_id,
                    "prompt": prompt,
                    "code": "",
                    "best_reward": 0.0,
                    "best_metadata": {"error_message": error_msg},
                    "total_turns": 1,
                    "turns_history": [],
                    "compilation_success": False,
                    "backtest_success": False,
                    "has_trades": False,
                    "total_trades": 0,
                    "error_message": error_msg,
                    "total_generation_time": total_time,
                    "model": self.model,
                    "max_turns": 1,
                    "generation_success": False,
                    "is_looped": False,
                    "max_repeats": 0,
                    "repeated_line": "",
                    "api_error": api_error
                }
            
            cleaned_code = self._clean_code(generated_code)
            
            if self.verbose:
                print(f"[STRATEGY {strategy_id}] Testing code ({yf_symbol}@{timeframe})...", flush=True)
            reward, metadata = await self._test_strategy_async(
                cleaned_code, strategy_id, prompt,
                yf_symbol=yf_symbol, timeframe=timeframe,
            )
            
            # Verbose result logging
            if self.verbose:
                comp = "✓" if metadata.get("compilation_success") else "✗"
                back = "✓" if metadata.get("backtest_success") else "✗"
                trades = "✓" if metadata.get("has_trades") else "✗"
                if metadata.get("judge_called"):
                    judge = "✓" if metadata.get("judge_aligned") else "✗"
                else:
                    judge = "-"
                print(f"[STRATEGY {strategy_id}] reward={reward} [comp:{comp} back:{back} trades:{trades} judge:{judge}]", flush=True)
            
            total_time = time.time() - start_time
            
            # Detect repetition loops
            is_looped, max_repeats, repeated_line = detect_repetition_loop(cleaned_code)
            if self.verbose and is_looped:
                print(f"[STRATEGY {strategy_id}] ⚠️ Looped detected: {max_repeats} repeats of '{repeated_line[:40]}'", flush=True)
            
            result = {
                "strategy_id": strategy_id,
                "prompt": prompt,
                "code": cleaned_code,
                "best_reward": reward,
                "best_metadata": metadata,
                "total_turns": 1,
                "turns_history": [{
                    "turn": 1,
                    "code": cleaned_code,
                    "reward": reward,
                    "metadata": metadata,
                    "feedback": "",
                    "generation_time": total_time
                }],
                "compilation_success": metadata.get("compilation_success", False),
                "backtest_success": metadata.get("backtest_success", False),
                "has_trades": metadata.get("has_trades", False),
                "total_trades": metadata.get("total_trades", 0),
                "error_message": metadata.get("error_message"),
                "total_generation_time": total_time,
                "model": self.model,
                "max_turns": 1,
                "generation_success": reward > 0.0,
                "is_looped": is_looped,
                "max_repeats": max_repeats,
                "repeated_line": repeated_line,
                "api_error": False
            }
            
            # Save trajectory
            self._write_trajectory(strategy_id, result)
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            if self.verbose:
                print(f"[STRATEGY {strategy_id}] Exception: {str(e)}", flush=True)
            result = {
                "strategy_id": strategy_id,
                "prompt": prompt,
                "code": "",
                "best_reward": 0.0,
                "best_metadata": {"error_message": str(e)},
                "total_turns": 1,
                "turns_history": [],
                "compilation_success": False,
                "backtest_success": False,
                "has_trades": False,
                "total_trades": 0,
                "error_message": str(e),
                "total_generation_time": total_time,
                "model": self.model,
                "max_turns": 1,
                "generation_success": False,
                "is_looped": False,
                "max_repeats": 0,
                "repeated_line": "",
                "api_error": True
            }
            
            # Save trajectory even on error
            self._write_trajectory(strategy_id, result)
            
            return result
    
    async def _generate_agentic_async(
        self,
        prompt: str,
        strategy_id: int,
        yf_symbol: str = "AAPL",
        timeframe: str = "1d",
    ) -> Dict[str, Any]:
        """Multi-turn agentic generation with feedback."""
        if self.verbose:
            print(f"[STRATEGY {strategy_id}] Starting agentic generation (max_turns={self.max_turns}, {yf_symbol}@{timeframe})", flush=True)
        start_time = time.time()
        
        turns_history = []
        best_reward = 0.0
        best_code = ""
        best_metadata = {}
        conversation = []
        
        # Open trajectory file for real-time logging
        trajectory_file = None
        session_id = str(uuid.uuid4())[:8]
        if self.trajectories_dir:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self.trajectories_dir,
                f"strategy_{strategy_id}_{timestamp_str}_{session_id}.jsonl"
            )
            trajectory_file = open(filepath, "w", encoding="utf-8")
            # Write session_start event
            trajectory_file.write(json.dumps({
                "event_type": "session_start",
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "task_id": strategy_id,
                "task_question": prompt[:500],
                "max_turns": self.max_turns,
                "mode": "agentic"
            }, ensure_ascii=False) + "\n")
            trajectory_file.flush()
        
        try:
            for turn in range(1, self.max_turns + 1):
                turn_start = time.time()
                
                try:
                    if turn == 1:
                        user_message = SYSTEM_PROMPT_EN + _USER_PROMPT_AGENTIC_EN.format(prompt=prompt)
                        conversation = [{"role": "user", "content": user_message}]
                    else:
                        prev = turns_history[-1]
                        conversation.append({"role": "assistant", "content": prev["code"]})
                        conversation.append({"role": "user", "content": prev["feedback"]})
                    
                    request_params = {
                        "model": self.model,
                        "messages": conversation,
                        "temperature": 0.1,
                        "max_tokens": self.max_tokens,
                    }
                    if self.extra_body:
                        request_params["extra_body"] = self.extra_body

                    response = await self.async_client.chat.completions.create(**request_params)
                    
                    generated_code = response.choices[0].message.content or ""
                    finish_reason = response.choices[0].finish_reason
                    
                    # Extract token counts from response
                    reasoning_tokens = 0
                    completion_tokens = 0
                    if hasattr(response, 'usage') and response.usage:
                        completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                        if hasattr(response.usage, 'completion_tokens_details'):
                            details = response.usage.completion_tokens_details
                            if details and hasattr(details, 'reasoning_tokens'):
                                reasoning_tokens = getattr(details, 'reasoning_tokens', 0) or 0
                    
                    if completion_tokens == 0 and generated_code:
                        completion_tokens = count_tokens(generated_code)
                    
                    if self.verbose:
                        print(f"[STRATEGY {strategy_id}] Turn {turn}/{self.max_turns}: {completion_tokens} tokens (reasoning: {reasoning_tokens}), finish_reason={finish_reason}", flush=True)
                    
                    cleaned_code = self._clean_code(generated_code)
                    
                    reward, metadata = await self._test_strategy_async(
                        cleaned_code, strategy_id, prompt,
                        yf_symbol=yf_symbol, timeframe=timeframe,
                    )
                    feedback = self._build_feedback(metadata, turn)
                    
                    # Verbose result logging
                    if self.verbose:
                        comp = "✓" if metadata.get("compilation_success") else "✗"
                        back = "✓" if metadata.get("backtest_success") else "✗"
                        trades = "✓" if metadata.get("has_trades") else "✗"
                        if metadata.get("judge_called"):
                            judge = "✓" if metadata.get("judge_aligned") else "✗"
                        else:
                            judge = "-"
                        print(f"[STRATEGY {strategy_id}] Turn {turn} result: reward={reward} [comp:{comp} back:{back} trades:{trades} judge:{judge}]", flush=True)
                    
                    turn_time = time.time() - turn_start
                    
                    turn_info = {
                        "turn": turn,
                        "code": cleaned_code,
                        "reward": reward,
                        "metadata": metadata,
                        "feedback": feedback,
                        "generation_time": turn_time
                    }
                    turns_history.append(turn_info)
                    
                    # Write turn_complete event in real-time
                    if trajectory_file:
                        trajectory_file.write(json.dumps({
                            "event_type": "turn_complete",
                            "timestamp": datetime.now().isoformat(),
                            "session_id": session_id,
                            "task_id": strategy_id,
                            "turn_number": turn,
                            "max_turns": self.max_turns,
                            "reward": float(reward),
                            "metadata": metadata,
                            "generated_code": cleaned_code,
                            "feedback": feedback,
                            "turn_time": turn_time
                        }, ensure_ascii=False) + "\n")
                        trajectory_file.flush()
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_code = cleaned_code
                        best_metadata = metadata
                    
                    # Stop on success
                    if reward == 1.0 and metadata.get("has_trades"):
                        break
                        
                except Exception as e:
                    turn_time = time.time() - turn_start
                    turn_info = {
                        "turn": turn,
                        "code": "",
                        "reward": 0.0,
                        "metadata": {"error_message": str(e)},
                        "feedback": f"API Error: {str(e)}",
                        "generation_time": turn_time
                    }
                    turns_history.append(turn_info)
                    
                    # Write error event
                    if trajectory_file:
                        trajectory_file.write(json.dumps({
                            "event_type": "turn_complete",
                            "timestamp": datetime.now().isoformat(),
                            "session_id": session_id,
                            "task_id": strategy_id,
                            "turn_number": turn,
                            "max_turns": self.max_turns,
                            "reward": 0.0,
                            "metadata": {"error_message": str(e)},
                            "generated_code": "",
                            "feedback": f"API Error: {str(e)}",
                            "turn_time": turn_time
                        }, ensure_ascii=False) + "\n")
                        trajectory_file.flush()
            
            total_time = time.time() - start_time
            
            # Write session_complete event
            if trajectory_file:
                trajectory_file.write(json.dumps({
                    "event_type": "session_complete",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "task_id": strategy_id,
                    "total_turns": len(turns_history),
                    "final_reward": float(best_reward),
                    "success": best_reward > 0.0,
                    "total_time": total_time
                }, ensure_ascii=False) + "\n")
        
        finally:
            if trajectory_file:
                trajectory_file.close()
        
        # Detect repetition loops in best code
        is_looped, max_repeats, repeated_line = detect_repetition_loop(best_code)
        if self.verbose and is_looped:
            print(f"[STRATEGY {strategy_id}] ⚠️ Looped detected: {max_repeats} repeats of '{repeated_line[:40]}'", flush=True)
        
        return {
            "strategy_id": strategy_id,
            "prompt": prompt,
            "code": best_code,
            "best_reward": best_reward,
            "best_metadata": best_metadata,
            "total_turns": len(turns_history),
            "turns_history": turns_history,
            "compilation_success": best_metadata.get("compilation_success", False),
            "backtest_success": best_metadata.get("backtest_success", False),
            "has_trades": best_metadata.get("has_trades", False),
            "total_trades": best_metadata.get("total_trades", 0),
            "error_message": best_metadata.get("error_message"),
            "total_generation_time": total_time,
            "model": self.model,
            "max_turns": self.max_turns,
            "generation_success": best_reward > 0.0,
            "is_looped": is_looped,
            "max_repeats": max_repeats,
            "repeated_line": repeated_line
        }
    
    async def generate_batch_async(
        self,
        dataset: List[Dict[str, Any]],
        batch_size: int = 20,
        delay_between_batches: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Generate strategies for a batch of tasks.
        
        Args:
            dataset: List of task dictionaries with 'reformulated_task' or 'prompt'
            batch_size: Number of parallel requests
            delay_between_batches: Delay between batches in seconds
        
        Returns:
            List of generation results
        """
        results = []
        total = len(dataset)
        
        generate_fn = (
            self._generate_single_async 
            if self.max_turns == 1 
            else self._generate_agentic_async
        )
        
        mode = "single-shot" if self.max_turns == 1 else f"agentic (max_turns={self.max_turns})"
        print(f"Starting {mode} generation for {total} strategies...")
        
        for i in range(0, total, batch_size):
            batch = dataset[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            print(f"Batch {batch_num}/{total_batches}: processing {len(batch)} strategies...")
            
            tasks = []
            for item in batch:
                prompt = item.get("reformulated_task") or item.get("prompt") or item.get("description", "")
                strategy_id = item.get("strategy_id", 0)
                yf_symbol = item.get("yf_symbol", "AAPL")
                timeframe = item.get("timeframe", "1d")
                tasks.append(generate_fn(prompt, strategy_id, yf_symbol=yf_symbol, timeframe=timeframe))
            
            batch_results = await asyncio.gather(*tasks)
            
            for result, item in zip(batch_results, batch):
                result["source"] = item.get("source", "unknown")
                result["difficulty"] = item.get("difficulty", "unknown")
                results.append(result)
            
            success_count = sum(1 for r in batch_results if r.get("has_trades", False))
            print(f"  Completed: {success_count}/{len(batch)} successful")
            
            if i + batch_size < total:
                await asyncio.sleep(delay_between_batches)
        
        return results
    
    def generate_batch(
        self,
        dataset: List[Dict[str, Any]],
        batch_size: int = 20,
        delay_between_batches: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for batch generation."""
        return asyncio.run(
            self.generate_batch_async(dataset, batch_size, delay_between_batches)
        )
    
    def generate_single(self, prompt: str, strategy_id: int = 0) -> Dict[str, Any]:
        """Generate a single strategy synchronously."""
        if self.max_turns == 1:
            return asyncio.run(self._generate_single_async(prompt, strategy_id))
        else:
            return asyncio.run(self._generate_agentic_async(prompt, strategy_id))


if __name__ == "__main__":
    # Test single-shot generation
    print("Testing single-shot generator...")
    generator = StrategyGenerator(max_turns=1)
    
    test_prompt = "Create a simple SMA crossover strategy with 10 and 30 period moving averages"
    result = generator.generate_single(test_prompt, strategy_id=1)
    
    print(f"Success: {result['generation_success']}")
    print(f"Turns: {result['total_turns']}")
    print(f"Trades: {result['total_trades']}")
    print(f"Time: {result['total_generation_time']:.2f}s")

