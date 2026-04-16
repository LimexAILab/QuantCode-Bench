#!/usr/bin/env python3
"""
Agentic benchmark runner.

Evaluates LLM ability to iteratively refine trading strategies with feedback.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from quantcode_bench import StrategyGenerator, preload_benchmark_data, preload_multiframe_data


def load_dataset(path: str) -> list:
    """Load evaluation dataset from JSON file."""
    print(f"Loading dataset from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} tasks")
    return data


def print_summary(results: list, max_turns: int) -> dict:
    """Print and return benchmark summary."""
    total = len(results)
    if total == 0:
        print("No results to summarize")
        return {}
    
    compilation_ok = sum(1 for r in results if r.get("compilation_success", False))
    backtest_ok = sum(1 for r in results if r.get("backtest_success", False))
    has_trades = sum(1 for r in results if r.get("has_trades", False))
    total_turns = sum(r.get("total_turns", 0) for r in results)
    empty_responses = sum(1 for r in results if not r.get("code"))
    api_errors = sum(1 for r in results if r.get("api_error", False))
    
    # Loop detection metrics
    looped_count = sum(1 for r in results if r.get("is_looped", False))
    looped_no_next = sum(1 for r in results if r.get("is_looped", False) and "def next(self)" not in r.get("code", ""))
    max_repeats_overall = max((r.get("max_repeats", 0) for r in results), default=0)
    
    # Judge metrics
    trades_before_judge = 0
    judge_passed = 0
    judge_rejected = 0
    for r in results:
        meta = r.get("best_metadata", {})
        if meta.get("has_trades"):
            trades_before_judge += 1
            if meta.get("judge_called"):
                if meta.get("judge_aligned"):
                    judge_passed += 1
                else:
                    judge_rejected += 1
            else:
                judge_passed += 1
    
    compilation_rate = compilation_ok / total * 100
    backtest_rate = backtest_ok / total * 100
    trade_rate = has_trades / total * 100
    avg_turns = total_turns / total
    trades_before_judge_rate = trades_before_judge / total * 100
    judge_pass_rate = judge_passed / total * 100
    looped_rate = looped_count / total * 100
    
    # Average turns for successful strategies
    successful = [r for r in results if r.get("has_trades")]
    avg_turns_success = sum(r.get("total_turns", 0) for r in successful) / len(successful) if successful else 0
    
    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY (Agentic Mode, max_turns={max_turns})")
    print("=" * 60)
    print(f"Total strategies:         {total}")
    print(f"Compilation success:      {compilation_ok} ({compilation_rate:.1f}%)")
    print(f"Backtest success:         {backtest_ok} ({backtest_rate:.1f}%)")
    print(f"Trades (before judge):    {trades_before_judge} ({trades_before_judge_rate:.1f}%)")
    print(f"Judge passed:             {judge_passed} ({judge_pass_rate:.1f}%)")
    print(f"Judge rejected:           {judge_rejected}")
    print(f"Empty responses:          {empty_responses}")
    print(f"⚠️  API errors (timeout):  {api_errors}")
    print(f"🔄 Looped count:          {looped_count} ({looped_rate:.1f}%)")
    print(f"🔄 Looped w/o next():     {looped_no_next} (max repeats: {max_repeats_overall})")
    print(f"\nIteration statistics:")
    print(f"  Total LLM calls:        {total_turns}")
    print(f"  Average turns/task:     {avg_turns:.2f}")
    if successful:
        print(f"  Avg turns (success):    {avg_turns_success:.2f}")
    
    # Distribution of turns to success
    print("\nTurns to success distribution:")
    turn_dist = {}
    for r in successful:
        t = r.get("total_turns", 0)
        turn_dist[t] = turn_dist.get(t, 0) + 1
    
    for turns in sorted(turn_dist.keys()):
        count = turn_dist[turns]
        print(f"  Turn {turns}: {count} strategies")
    
    # Stats by difficulty
    print("\nBy Difficulty:")
    difficulties = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        if diff not in difficulties:
            difficulties[diff] = {"total": 0, "success": 0, "turns": 0}
        difficulties[diff]["total"] += 1
        difficulties[diff]["turns"] += r.get("total_turns", 0)
        if r.get("has_trades"):
            difficulties[diff]["success"] += 1
    
    for diff, stats in sorted(difficulties.items()):
        rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg = stats["turns"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {diff}: {stats['success']}/{stats['total']} ({rate:.1f}%), avg_turns={avg:.2f}")
    
    return {
        "total": total,
        "compilation_rate": compilation_rate,
        "backtest_rate": backtest_rate,
        "trades_before_judge_rate": trades_before_judge_rate,
        "judge_pass_rate": judge_pass_rate,
        "judge_rejected": judge_rejected,
        "empty_responses": empty_responses,
        "api_errors": api_errors,
        "looped_count": looped_count,
        "looped_rate": looped_rate,
        "looped_no_next": looped_no_next,
        "max_repeats": max_repeats_overall,
        "trade_rate": trade_rate,
        "avg_turns": avg_turns,
        "avg_turns_success": avg_turns_success,
        "total_llm_calls": total_turns,
        "by_difficulty": difficulties
    }


def main():
    parser = argparse.ArgumentParser(description="Run agentic benchmark")
    parser.add_argument("--dataset", default="data/benchmark_tasks_multiframe.json", help="Path to dataset JSON")
    parser.add_argument("--model", default=None, help="Model name (or set GENERATOR_MODEL)")
    parser.add_argument("--base-url", default=None, help="API base URL (or set GENERATOR_BASE_URL)")
    parser.add_argument("--api-key", default=None, help="API key (or set GENERATOR_API_KEY)")
    parser.add_argument("--max-turns", type=int, default=10, help="Max iterations per strategy")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--batch-size", type=int, default=20, help="Parallel batch size")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--save-trajectories", action="store_true", help="Save individual trajectories as JSONL files")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable thinking mode (for sglang with reasoning models)")
    parser.add_argument("--max-tokens", type=int, default=64000, help="Max tokens for generation response")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    # Preload data cache for all unique (symbol, interval) pairs
    print("\nPreloading market data...")
    preload_multiframe_data(dataset)
    
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"Limited to {args.limit} tasks")
    
    # Add strategy IDs
    for idx, item in enumerate(dataset, 1):
        item["strategy_id"] = idx
    
    # Determine model name for trajectory directory
    model_name = args.model or os.environ.get("GENERATOR_MODEL", "unknown")
    model_name_safe = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    
    # Create trajectories directory if needed (with model name)
    trajectories_dir = None
    if args.save_trajectories:
        trajectories_dir = os.path.join(args.output_dir, f"trajectories_agentic_{model_name_safe}_{timestamp}")
        os.makedirs(trajectories_dir, exist_ok=True)
        print(f"Trajectories will be saved to: {trajectories_dir}")
    
    # Build extra_body for sglang thinking control
    extra_body = None
    if args.disable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    
    # Initialize generator
    generator = StrategyGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_turns=args.max_turns,  # Agentic mode
        trajectories_dir=trajectories_dir,
        extra_body=extra_body,
        max_tokens=args.max_tokens,
    )
    
    print(f"\nModel: {generator.model}")
    print(f"Mode: Agentic (max_turns={args.max_turns})")
    print(f"Batch size: {args.batch_size}")
    
    # Run benchmark
    print("\n" + "=" * 60)
    print("STARTING AGENTIC BENCHMARK")
    print("=" * 60)
    
    start_time = time.time()
    results = generator.generate_batch(dataset, batch_size=args.batch_size)
    total_time = time.time() - start_time
    
    # Print summary
    summary = print_summary(results, args.max_turns)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"agentic_{model_name_safe}_{timestamp}.json")
    output_data = {
        "benchmark_info": {
            "mode": "agentic",
            "max_turns": args.max_turns,
            "timestamp": timestamp,
            "model": generator.model,
            "total_tasks": len(results),
            "total_time_seconds": total_time
        },
        "summary": summary,
        "results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    if len(results) > 0:
        print(f"Average time per strategy: {total_time/len(results):.2f}s")
        print(f"Total LLM calls: {summary.get('total_llm_calls', 0)}")


if __name__ == "__main__":
    main()

