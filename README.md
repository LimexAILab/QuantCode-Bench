# QuantCode-Bench

A benchmark for evaluating the ability of large language models to generate executable algorithmic trading strategies for [Backtrader](https://www.backtrader.com/).

## Overview

QuantCode-Bench measures how well LLMs can translate natural-language strategy descriptions into functional trading code. Unlike standard code benchmarks, trading-strategy generation requires simultaneous mastery of domain-specific financial logic, knowledge of a specialized API, and the ability to produce code that is not only syntactically correct but also leads to actual trades on historical data.

Each task specifies its own ticker and timeframe (e.g. AAPL daily, BTC-USD 1h, EURUSD=X 15m), so the model must produce strategies that work across diverse market instruments and data granularities.

The benchmark uses a **four-stage evaluation pipeline**:

1. **Compilation** -- the code is syntactically correct
2. **Backtest** -- the strategy executes on asset-specific historical data without runtime errors
3. **Trade** -- the strategy places at least one trade
4. **Judge** -- an LLM judge confirms semantic alignment with the task description

Two evaluation settings are supported:

- **Single-turn**: the model must generate a correct strategy on the first attempt
- **Agentic multi-turn**: the model receives structured feedback and may iteratively refine the strategy (up to 10 turns)

## Dataset

The benchmark contains **400 trading-strategy generation tasks** collected from multiple sources:

| Source | Count |
|--------|-------|
| Reddit | 183 |
| TradingView | 100 |
| StackExchange | 90 |
| GitHub | 19 |
| Synthetic | 8 |

Tasks span three difficulty levels:

| Difficulty | Count |
|------------|-------|
| Easy | 197 |
| Medium | 116 |
| Hard | 87 |

Each task includes a per-task ticker and timeframe so that strategies are backtested on the appropriate market data:

```json
{
  "id": 42,
  "reformulated_task": "Create a strategy that buys when RSI drops below 30 and sells when RSI rises above 70",
  "source": "reddit",
  "difficulty": "easy",
  "ticker": "generic",
  "yf_symbol": "AAPL",
  "timeframe": "1d"
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Caching

Before running the benchmark, pre-download market data for all ticker/timeframe pairs to avoid rate-limiting during evaluation:

```bash
python scripts/build_cache.py
```

This reads `data/task_data_requirements.json` and downloads OHLCV data from yfinance into `data/cache/` as pickle files. Cached data is automatically loaded during backtest execution.

## Quick Start

### Single-Turn Benchmark

```bash
export GENERATOR_BASE_URL="https://openrouter.ai/api/v1"
export GENERATOR_API_KEY="your-api-key"
export GENERATOR_MODEL="anthropic/claude-sonnet-4"

python run_single_shot.py --dataset data/benchmark_tasks_multiframe.json --limit 100
```

### Agentic Multi-Turn Benchmark

```bash
python run_agentic.py --dataset data/benchmark_tasks_multiframe.json --max-turns 10 --limit 100
```

## Configuration

### Generator

```bash
export GENERATOR_BASE_URL="https://openrouter.ai/api/v1"
export GENERATOR_API_KEY="your-api-key"
export GENERATOR_MODEL="anthropic/claude-sonnet-4"
```

### Judge (optional, uses generator settings if not set)

```bash
export JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export JUDGE_API_KEY="your-api-key"
export JUDGE_MODEL="anthropic/claude-sonnet-4"
export JUDGE_ENABLED="true"
export JUDGE_MODE="api"  # "api" (default) or "sglang" for local models
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Compilation Rate | Percentage of strategies that compile without errors |
| Backtest Rate | Percentage of strategies that execute without runtime errors |
| Trade Rate | Percentage of strategies that generate at least one trade |
| Judge Pass Rate | Percentage of strategies that pass semantic alignment check |

## Project Structure

```
QuantCode-Bench/
├── quantcode_bench/
│   ├── __init__.py          # Package exports
│   ├── reward.py            # Reward function with execution pipeline
│   ├── judge.py             # LLM-based strategy alignment judge
│   ├── generator.py         # Strategy generator (single-shot & agentic)
│   └── data_cache.py        # Market data caching
├── data/
│   ├── benchmark_tasks_multiframe.json  # 400 evaluation tasks
│   └── task_data_requirements.json      # Per-task data requirements
├── scripts/
│   ├── build_cache.py       # Download & cache market data
│   └── run_all_models.sh    # Batch benchmark runner
├── examples/
│   └── sma_crossover.py     # Example strategy
├── run_single_shot.py       # Single-turn benchmark runner
├── run_agentic.py           # Agentic benchmark runner
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use QuantCode-Bench in your research, please cite:

```bibtex
@article{khoroshilov2026quantcodebench,
  title={QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies},
  author={Khoroshilov Alexey and Chernysh Alexey and Ekhtibarov Orkhan and Kamkia Nini and Zmitrovich Dmitry},
  year={2026},
  url={https://github.com/LimexAILab/QuantCode-Bench}
}
```

## License

[MIT License](LICENSE)
