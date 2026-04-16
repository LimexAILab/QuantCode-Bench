"""
Reward function for evaluating generated trading strategies.

Pipeline: Code Cleaning -> Validation -> Backtest Execution -> Judge Evaluation

Returns binary reward:
- 1.0: Strategy compiles, runs, generates trades, and passes judge
- 0.0: Any step fails
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Optional, Tuple

from .judge import create_strategy_judge, StrategyJudge

logger = logging.getLogger(__name__)

# Global state for caching
_data_cache: Dict[str, str] = {}  # (symbol, interval) -> temp file path
_cache_lock: Optional[threading.Lock] = None
_judge_instance: Optional[StrategyJudge] = None
_judge_enabled = True

_PROJECT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")


def _get_or_create_data_cache(
    symbol: str = "AAPL",
    interval: str = "1d",
) -> str:
    """Get or create cached data file for backtesting.

    Lookup order:
    1. In-memory registry of temp file paths (already loaded this run)
    2. Pre-built pickle in ``data/cache/{symbol}_{interval}.pkl``
    3. Fallback: download from yfinance and cache to temp dir
    """
    global _cache_lock, _data_cache

    if _cache_lock is None:
        _cache_lock = threading.Lock()

    cache_key = f"{symbol}_{interval}"

    if cache_key in _data_cache and os.path.exists(_data_cache[cache_key]):
        return _data_cache[cache_key]

    with _cache_lock:
        if cache_key in _data_cache and os.path.exists(_data_cache[cache_key]):
            return _data_cache[cache_key]

        import pickle

        safe_name = symbol.replace("=", "_").replace("^", "_")
        project_pkl = os.path.join(_PROJECT_CACHE_DIR, f"{safe_name}_{interval}.pkl")
        if os.path.exists(project_pkl):
            tmp_path = os.path.join(tempfile.gettempdir(), f"qcb_{safe_name}_{interval}.pkl")
            if not os.path.exists(tmp_path):
                import shutil
                shutil.copy2(project_pkl, tmp_path)
            logger.info(f"Using project cache: {project_pkl} ({safe_name}@{interval})")
            _data_cache[cache_key] = tmp_path
            return tmp_path

        tmp_path = os.path.join(tempfile.gettempdir(), f"qcb_{safe_name}_{interval}.pkl")
        if os.path.exists(tmp_path):
            logger.info(f"Using temp cache: {tmp_path}")
            _data_cache[cache_key] = tmp_path
            return tmp_path

        import yfinance as yf

        logger.info(f"Downloading {symbol}@{interval} from yfinance...")
        ticker = yf.Ticker(symbol)
        if interval == "1d":
            df = ticker.history(start="2020-01-01", end="2025-12-31")
        elif interval == "1h":
            df = ticker.history(period="730d", interval="1h")
        else:
            period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d"}
            df = ticker.history(period=period_map.get(interval, "60d"), interval=interval)

        if df is None or df.empty:
            raise ValueError(f"No data for {symbol}@{interval}")

        with open(tmp_path, "wb") as f:
            pickle.dump(df, f)

        logger.info(f"Cached {len(df)} rows to {tmp_path} ({symbol}@{interval})")
        _data_cache[cache_key] = tmp_path
        return tmp_path


def _get_or_create_judge() -> Optional[StrategyJudge]:
    """Get or create the strategy judge instance."""
    global _judge_instance, _judge_enabled
    
    judge_enabled_env = os.environ.get("JUDGE_ENABLED", "true").lower()
    if judge_enabled_env in ["false", "0", "no"]:
        _judge_enabled = False
        return None
    
    if _judge_instance is not None:
        return _judge_instance
    
    try:
        _judge_instance = create_strategy_judge()
        return _judge_instance
    except Exception as e:
        logger.error(f"Failed to initialize judge: {e}")
        _judge_enabled = False
        return None


def _clean_code(code: str) -> str:
    """Clean generated code from markdown, thinking blocks, and extra text."""
    if not code:
        return ""
    
    # Remove thinking/reasoning blocks
    if "</think>" in code:
        last_end = code.rfind("</think>")
        code = code[last_end + len("</think>"):].strip()
    elif "<think>" in code:
        # Find code after think block
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ", "class ", "def ")):
                code = "\n".join(lines[i:])
                break
        else:
            code = ""
    
    # Remove markdown code blocks
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


def _validate_code_structure(code: str) -> Tuple[bool, str]:
    """Validate basic code structure requirements."""
    if not code:
        return False, "Empty code"
    
    if "import backtrader" not in code:
        return False, "Missing 'import backtrader' statement"
    
    if not re.search(r"class\s+\w+\s*\(\s*bt\.Strategy\s*\)", code):
        return False, "Missing strategy class inheriting from bt.Strategy"
    
    if "def next(self)" not in code:
        return False, "Missing next() method"
    
    return True, ""


def _create_test_wrapper(strategy_code: str, data_file_path: str) -> str:
    """Create test execution wrapper for the strategy."""
    return f'''
{strategy_code}

import sys
import json
import pickle
import backtrader as bt

def run_backtest():
    """Execute backtest and return results."""
    
    # Find strategy class
    strategy_class = None
    for name in list(globals().keys()):
        try:
            obj = globals()[name]
            if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj != bt.Strategy:
                strategy_class = obj
                break
        except (TypeError, AttributeError):
            continue
    
    if strategy_class is None:
        return {{"success": False, "error": "No strategy class found"}}
    
    # Load data
    try:
        with open("{data_file_path}", "rb") as f:
            df = pickle.load(f)
        if df.empty:
            return {{"success": False, "error": "Empty data"}}
    except Exception as e:
        return {{"success": False, "error": f"Data loading error: {{str(e)}}"}}
    
    # Run backtest
    try:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class)
        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(10000)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()
        
        strat = results[0]
        trades = strat.analyzers.trades.get_analysis()
        total_trades = trades.get("total", {{}}).get("total", 0) if trades else 0
        
        return {{
            "success": True,
            "has_trades": total_trades > 0,
            "total_trades": total_trades,
            "start_value": start_value,
            "end_value": end_value,
            "total_return": (end_value - start_value) / start_value * 100 if start_value > 0 else 0
        }}
    except Exception as e:
        import traceback
        return {{"success": False, "error": f"Backtest error: {{str(e)}}", "traceback": traceback.format_exc()}}

result = run_backtest()
print(json.dumps(result))
'''


def _execute_strategy(
    strategy_code: str,
    symbol: str = "AAPL",
    interval: str = "1d",
) -> Dict[str, Any]:
    """Execute strategy and return results."""
    start_time = time.time()
    
    try:
        data_file_path = _get_or_create_data_cache(symbol=symbol, interval=interval)
        test_code = _create_test_wrapper(strategy_code, data_file_path)
        
        process = subprocess.Popen(
            [sys.executable, "-c", test_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(timeout=120)
        execution_time = time.time() - start_time
        
        if process.returncode != 0:
            error_lines = stderr.strip().split("\n")[-10:]
            return {
                "compilation_success": True,
                "backtest_success": False,
                "has_trades": False,
                "error_message": "\n".join(error_lines),
                "total_trades": 0,
                "execution_time": execution_time
            }
        
        # Parse JSON output
        for line in reversed(stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                result = json.loads(line)
                
                if result.get("success", False):
                    return {
                        "compilation_success": True,
                        "backtest_success": True,
                        "has_trades": result.get("has_trades", False),
                        "total_trades": result.get("total_trades", 0),
                        "total_return": result.get("total_return", 0),
                        "error_message": None,
                        "execution_time": execution_time
                    }
                else:
                    return {
                        "compilation_success": True,
                        "backtest_success": False,
                        "has_trades": False,
                        "error_message": result.get("error", "Unknown error"),
                        "total_trades": 0,
                        "execution_time": execution_time
                    }
        
        return {
            "compilation_success": True,
            "backtest_success": False,
            "has_trades": False,
            "error_message": "Could not parse output",
            "total_trades": 0,
            "execution_time": execution_time
        }
        
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return {
            "compilation_success": True,
            "backtest_success": False,
            "has_trades": False,
            "error_message": "Timeout: execution took >120s",
            "total_trades": 0,
            "execution_time": 120.0
        }
        
    except Exception as e:
        return {
            "compilation_success": False,
            "backtest_success": False,
            "has_trades": False,
            "error_message": f"Execution error: {str(e)}",
            "total_trades": 0,
            "execution_time": time.time() - start_time
        }


def backtest_reward_fn(task_info: Dict, code: str) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a generated trading strategy.
    
    Pipeline:
    1. Clean code (remove markdown, thinking blocks)
    2. Validate structure
    3. Execute backtest
    4. Judge alignment (if enabled and strategy works)
    
    Args:
        task_info: Dict with 'question' (task description) and optional 'strategy_id'
        code: Generated strategy code
    
    Returns:
        Tuple of (reward, metadata)
        - reward: 1.0 if all checks pass, 0.0 otherwise
        - metadata: Detailed execution information
    """
    strategy_id = task_info.get("strategy_id", 0)
    task_description = task_info.get("question", "")
    symbol = task_info.get("yf_symbol", "AAPL")
    interval = task_info.get("timeframe", "1d")
    
    # Step 1: Clean code
    cleaned_code = _clean_code(code)
    
    # Step 2: Validate structure
    is_valid, validation_error = _validate_code_structure(cleaned_code)
    
    if not is_valid:
        return 0.0, {
            "compilation_success": False,
            "backtest_success": False,
            "has_trades": False,
            "judge_aligned": False,
            "error_message": f"Validation failed: {validation_error}",
            "total_trades": 0,
            "strategy_id": strategy_id,
            "data_symbol": symbol,
            "data_interval": interval,
        }
    
    # Step 3: Execute backtest
    result = _execute_strategy(cleaned_code, symbol=symbol, interval=interval)
    
    # Step 4: Judge evaluation (only if strategy works)
    judge_aligned = True
    judge_called = False
    judge_metadata = {}
    
    compilation_ok = result["compilation_success"]
    backtest_ok = result["backtest_success"]
    has_trades = result["has_trades"]
    
    if _judge_enabled and task_description and compilation_ok and backtest_ok and has_trades:
        judge_called = True
        judge = _get_or_create_judge()
        
        if judge is not None:
            try:
                judge_aligned, _, judge_metadata = judge.evaluate(task_description, cleaned_code)
            except Exception as e:
                logger.error(f"Judge error: {e}")
                judge_aligned = True  # Don't penalize for judge failures
    
    # Calculate reward
    if compilation_ok and backtest_ok and has_trades and judge_aligned:
        reward = 1.0
    else:
        reward = 0.0
    
    # Build metadata
    metadata = {
        "reward": reward,
        "compilation_success": compilation_ok,
        "backtest_success": backtest_ok,
        "has_trades": has_trades,
        "total_trades": result.get("total_trades", 0),
        "judge_enabled": _judge_enabled,
        "judge_called": judge_called,
        "judge_aligned": judge_aligned,
        "error_message": result.get("error_message"),
        "strategy_id": strategy_id,
        "execution_time": result.get("execution_time"),
        "data_symbol": symbol,
        "data_interval": interval,
    }
    
    if result.get("total_return") is not None:
        metadata["total_return"] = result["total_return"]
    
    if judge_metadata:
        metadata.update(judge_metadata)
    
    return reward, metadata


if __name__ == "__main__":
    # Test the reward function
    logging.basicConfig(level=logging.INFO)
    
    test_code = """
import backtrader as bt

class TradingStrategy(bt.Strategy):
    params = (('fast', 10), ('slow', 30))
    
    def __init__(self):
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.params.slow)
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
    
    task_info = {
        "question": "Create SMA crossover strategy with 10 and 30 period moving averages",
        "strategy_id": 1
    }
    
    print("Testing reward function...")
    reward, metadata = backtest_reward_fn(task_info, test_code)
    print(f"Reward: {reward}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")

