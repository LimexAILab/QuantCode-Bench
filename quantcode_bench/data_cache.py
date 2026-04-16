"""
Market data caching utilities for efficient benchmark execution.

Caches yfinance data to avoid repeated API calls during strategy evaluation.
"""

import os
import pickle
from typing import Optional

import pandas as pd
import yfinance as yf


class DataCache:
    """Cache for market data from yfinance."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory for cached files. Defaults to .cache in current directory.
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, symbol: str, start_date: str, end_date: str) -> str:
        """Get the cache file path for a given symbol and date range."""
        filename = f"{symbol}_{start_date}_{end_date}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def load_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load market data with caching.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: Force reload from yfinance, ignoring cache
        
        Returns:
            DataFrame with OHLCV data
        
        Raises:
            ValueError: If no data could be loaded for the symbol
        """
        cache_path = self.get_cache_path(symbol, start_date, end_date)
        
        # Try loading from cache
        if not force_refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                print(f"Loaded {len(data)} rows from cache: {symbol}")
                return data
            except Exception as e:
                print(f"Cache read error: {e}, fetching fresh data")
        
        # Fetch from yfinance
        print(f"Fetching data from yfinance: {symbol} ({start_date} to {end_date})")
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Cached {len(data)} rows for {symbol}")
        except Exception as e:
            print(f"Cache write error: {e}")
        
        return data
    
    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            print("Cache cleared")


def preload_benchmark_data(
    symbol: str = "AAPL",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01"
) -> DataCache:
    """
    Preload market data for benchmark execution (legacy single-data mode).
    
    Args:
        symbol: Ticker symbol to preload
        start_date: Start date
        end_date: End date
    
    Returns:
        DataCache instance with preloaded data
    """
    cache = DataCache()
    data = cache.load_data(symbol, start_date, end_date)
    
    print(f"Preloaded {len(data)} rows for {symbol} ({start_date} to {end_date})")
    
    return cache


def preload_multiframe_data(tasks: list) -> None:
    """
    Preload all unique (symbol, interval) data pairs from task list.

    This triggers ``reward._get_or_create_data_cache`` for each pair so that
    the data is ready in the temp directory before backtests start.

    Args:
        tasks: List of task dicts with ``yf_symbol`` and ``timeframe`` fields.
    """
    from .reward import _get_or_create_data_cache

    pairs = set()
    for t in tasks:
        sym = t.get("yf_symbol", "AAPL")
        tf = t.get("timeframe", "1d")
        pairs.add((sym, tf))

    print(f"Preloading {len(pairs)} unique (symbol, interval) data pairs...")
    for sym, tf in sorted(pairs):
        try:
            path = _get_or_create_data_cache(symbol=sym, interval=tf)
            with open(path, "rb") as f:
                df = pickle.load(f)
            print(f"  {sym:>12s} @ {tf:>4s}: {len(df):>6d} bars")
        except Exception as e:
            print(f"  {sym:>12s} @ {tf:>4s}: FAILED ({e})")
    print("Preloading complete.")


if __name__ == "__main__":
    # Test data caching
    cache = preload_benchmark_data()
    print(f"Cache directory: {cache.cache_dir}")

