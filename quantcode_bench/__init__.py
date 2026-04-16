"""
QuantCode-Bench: A benchmark for evaluating LLM trading strategy generation.

Supports two evaluation modes:
- Single-shot: Model must succeed on first attempt
- Agentic: Model can iterate with feedback
"""

__version__ = "1.0.0"

from .reward import backtest_reward_fn
from .judge import StrategyJudge, create_strategy_judge
from .generator import StrategyGenerator
from .data_cache import DataCache, preload_benchmark_data, preload_multiframe_data

__all__ = [
    "backtest_reward_fn",
    "StrategyJudge",
    "create_strategy_judge",
    "StrategyGenerator",
    "DataCache",
    "preload_benchmark_data",
    "preload_multiframe_data",
]

