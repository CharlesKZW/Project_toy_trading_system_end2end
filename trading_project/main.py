"""
main.py
~~~~~~~

Entry point for running the end‑to‑end trading project.  This script
downloads or generates intraday market data, cleans it, constructs
features, runs a momentum trading strategy through the backtest
framework and prints performance statistics.  The resulting code
demonstrates a self‑contained workflow for research, strategy
development, backtesting and (optionally) live trading integration.

Usage:

    python main.py --symbol AAPL --days 5 --freq 1m

The script will produce a ``market_data.csv`` file in the working
directory and write an order log to ``order_log.csv``.  Adjust the
command line arguments to tailor the data period and frequency.
"""

import argparse
import os
import sys

import pandas as pd

from download_data import main as download_main
from strategy import MeanReversionStrategy
from backtester import Backtester


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data and add derived features.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw intraday OHLCV data with a ``Datetime`` column.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame indexed by ``Datetime`` with added
        ``return``, ``ma_fast`` and ``ma_slow`` columns.
    """
    df = df.copy()
    # Ensure numeric types for price/volume; coerce bad rows to NaN then drop
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Remove missing/duplicate rows after type coercion
    df = df.dropna().drop_duplicates()
    # ensure Datetime column exists and is datetime type
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    # sort chronologically
    df = df.sort_values("Datetime")
    # compute returns
    df["return"] = df["Close"].pct_change().fillna(0)
    # moving averages (fast=20, slow=60 bars)
    df["ma_fast"] = df["Close"].rolling(20, min_periods=1).mean()
    df["ma_slow"] = df["Close"].rolling(60, min_periods=1).mean()
    # do not set Datetime as index here; downstream components expect column
    return df


def run_backtest(symbol: str, days: int, freq: str, lookback_days: int,
                 upper_pct: float, lower_pct: float, flat_upper: float, flat_lower: float) -> None:
    # Step 1: Download or generate data
    download_args = ["--symbol", symbol, "--outfile", "market_data.csv", "--days", str(days), "--freq", freq]
    download_main(download_args)
    # Step 2: Load and clean data
    df = pd.read_csv("market_data.csv")
    df_clean = clean_and_engineer_features(df)
    # Step 3: Create strategy instance
    lookback_bars = lookback_days * 390  # approximate 390 minutes per trading day
    strategy = MeanReversionStrategy(
        lookback_bars=lookback_bars,
        upper_pct=upper_pct,
        lower_pct=lower_pct,
        flat_upper=flat_upper,
        flat_lower=flat_lower,
    )
    # Step 4: Run backtest
    backtester = Backtester(strategy=strategy, data=df_clean, order_size=10, initial_capital=100_000.0)
    backtester.run()
    results = backtester.results()
    print("Backtest results for", symbol)
    for key, value in results.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Run the trading backtest pipeline.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol")
    parser.add_argument("--days", type=int, default=5, help="Number of trading days")
    parser.add_argument("--freq", default="1m", help="Bar frequency (e.g., 1m, 5m)")
    parser.add_argument("--lookback-days", type=int, default=3, help="Rolling window in days for percentiles")
    parser.add_argument("--upper-pct", type=float, default=0.95, help="Upper percentile to trigger short")
    parser.add_argument("--lower-pct", type=float, default=0.05, help="Lower percentile to trigger long")
    parser.add_argument("--flat-upper", type=float, default=0.65, help="Upper percentile to flatten short")
    parser.add_argument("--flat-lower", type=float, default=0.35, help="Lower percentile to flatten long")
    args = parser.parse_args()
    run_backtest(
        args.symbol,
        args.days,
        args.freq,
        lookback_days=args.lookback_days,
        upper_pct=args.upper_pct,
        lower_pct=args.lower_pct,
        flat_upper=args.flat_upper,
        flat_lower=args.flat_lower,
    )


if __name__ == "__main__":
    main()
