#!/usr/bin/env python3
"""Download and cache all needed market data from yfinance with rate-limiting."""

import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd

REQS_FILE = Path(__file__).parent.parent / "data" / "task_data_requirements.json"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"

INTERVAL_CONFIG = {
    "1m":  {"period": "7d",  "start": None, "end": None},
    "5m":  {"period": "60d", "start": None, "end": None},
    "15m": {"period": "60d", "start": None, "end": None},
    "30m": {"period": "60d", "start": None, "end": None},
    "1h":  {"period": "730d", "start": None, "end": None},
    "1d":  {"period": None, "start": "2020-01-01", "end": "2025-12-31"},
}

RATE_LIMIT_SLEEP = 2.0


def download_pair(symbol: str, interval: str) -> pd.DataFrame | None:
    """Download data for a single (symbol, interval) pair."""
    cfg = INTERVAL_CONFIG.get(interval)
    if cfg is None:
        print(f"  WARNING: Unknown interval {interval}, skipping")
        return None

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            if cfg["period"]:
                df = ticker.history(period=cfg["period"], interval=interval)
            else:
                df = ticker.history(start=cfg["start"], end=cfg["end"], interval=interval)

            if df is not None and len(df) > 0:
                return df

            print(f"  Empty result for {symbol}@{interval} (attempt {attempt+1})")
        except Exception as e:
            print(f"  Error downloading {symbol}@{interval} (attempt {attempt+1}): {e}")

        time.sleep(RATE_LIMIT_SLEEP * (attempt + 1))

    return None


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(REQS_FILE) as f:
        reqs = json.load(f)

    pairs = set()
    for r in reqs:
        if r.get("data_available", True):
            pairs.add((r["yf_symbol"], r["timeframe"]))

    pairs_sorted = sorted(pairs)
    print(f"Need to download {len(pairs_sorted)} unique (symbol, interval) pairs")

    success = 0
    failed = []

    for i, (symbol, interval) in enumerate(pairs_sorted, 1):
        cache_file = CACHE_DIR / f"{symbol.replace('=', '_').replace('^', '_')}_{interval}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            print(f"[{i}/{len(pairs_sorted)}] {symbol:>12s} @ {interval:>4s}: CACHED ({len(df)} bars)")
            success += 1
            continue

        print(f"[{i}/{len(pairs_sorted)}] {symbol:>12s} @ {interval:>4s}: downloading...", end=" ", flush=True)
        df = download_pair(symbol, interval)

        if df is not None and len(df) > 0:
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
            print(f"OK ({len(df)} bars, {df.index[0].date()} to {df.index[-1].date()})")
            success += 1
        else:
            print("FAILED")
            failed.append((symbol, interval))

        time.sleep(RATE_LIMIT_SLEEP)

    print(f"\n{'='*60}")
    print(f"Cache build complete: {success}/{len(pairs_sorted)} pairs downloaded")
    if failed:
        print(f"Failed pairs ({len(failed)}):")
        for sym, tf in failed:
            print(f"  {sym} @ {tf}")

    total_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*.pkl"))
    print(f"Total cache size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
