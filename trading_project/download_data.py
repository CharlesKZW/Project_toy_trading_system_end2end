"""
download_data.py
~~~~~~~~~~~~~~~~~

This module downloads historical intraday market data for a given equity using
the ``yfinance`` library.  If the library is unavailable or the download fails,
the script exits with an error instead of generating synthetic data.

The resulting dataset is written to ``market_data.csv`` in the repository root.
The CSV contains the following columns: ``Datetime``, ``Open``, ``High``,
``Low``, ``Close`` and ``Volume``.  Datetime values are in ISO‑8601 format and
reflect the America/New_York time zone.

Usage:

    python download_data.py --symbol AAPL --outfile market_data.csv

"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd


def download_with_yfinance(symbol: str, days: int, freq: str) -> Optional[pd.DataFrame]:
    """Attempt to download data using yfinance.

    yfinance is unofficial and may not be available in restricted environments.
    This function tries to import yfinance and return a dataframe if successful.
    Otherwise it returns None.

    Parameters
    ----------
    symbol : str
        Ticker symbol to download.
    days : int
        Number of days to retrieve.
    freq : str
        Bar frequency (e.g., '1m', '5m').

    Returns
    -------
    pandas.DataFrame or None
        Dataframe with intraday bars or ``None`` if download fails.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return None
    try:
        data = yf.download(tickers=symbol, period=f"{days}d", interval=freq)
        if data.empty:
            return None
        data.reset_index(inplace=True) 
        return data[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download intraday data via yfinance.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--outfile", default="market_data.csv", help="Output CSV filename")
    parser.add_argument("--days", type=int, default=7, help="Number of days to retrieve")
    parser.add_argument("--freq", default="1m", help="Bar frequency (default: 1m)")
    options = parser.parse_args(args)

    df: Optional[pd.DataFrame] = None

    # First try to load existing data file if present
    if os.path.exists(options.outfile):
        try:
            existing = pd.read_csv(options.outfile)
            required_cols = {"Datetime", "Open", "High", "Low", "Close", "Volume"}
            if required_cols.issubset(existing.columns):
                df = existing
                print(f"Loaded existing data from {options.outfile} ({len(df)} rows).")
            else:
                print(f"Existing file {options.outfile} missing required columns; will attempt download.")
        except Exception as exc:
            print(f"Failed to read existing file {options.outfile}: {exc}. Will attempt download.")

    # download data; abort if unsuccessful
    if df is None:
        df = download_with_yfinance(options.symbol, options.days, options.freq)
        if df is None:
            print(
                "Failed to download data using yfinance. "
                "Ensure yfinance is installed and network access is available.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Downloaded {len(df)} rows of data for {options.symbol} using yfinance.")
        df.to_csv(options.outfile, index=False)
        print(f"Data saved to {options.outfile}")


if __name__ == "__main__":
    main()
