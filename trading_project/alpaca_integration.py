"""
alpaca_integration.py
~~~~~~~~~~~~~~~~~~~~~

This script demonstrates how to connect to Alpaca’s paper trading API
using the official Python SDK, download recent minute‑level bar data for
an equity, save the data to disk, generate trading signals using
``MomentumStrategy``, and optionally place paper market orders that
follow those signals.  Order submission is **paper only** via the paper
endpoint.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

from strategy import MeanReversionStrategy


CONFIG_PATH = Path(__file__).resolve().parent / "alpaca_config.json"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Create it with your Alpaca settings."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["api_key", "api_secret", "base_url", "symbol", "start_date", "end_date"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required config keys in {path}: {missing}")
    return cfg

def get_client(cfg: Dict[str, Any]) -> tradeapi.REST:
    """Create an Alpaca SDK client configured for paper trading."""
    return tradeapi.REST(cfg["api_key"], cfg["api_secret"], cfg["base_url"], api_version="v2")


def _to_utc_iso(ts_str: str, default_time: str = "00:00:00") -> str:
    """Convert a date or datetime string to RFC3339 UTC for Alpaca."""
    ts = pd.to_datetime(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("UTC")
    # If only date provided, set time
    if len(ts_str) <= 10:
        ts = ts.normalize() + pd.to_timedelta(default_time)
    return ts.isoformat()


def fetch_bars(client: tradeapi.REST, symbol: str, start: str, end: str,
               timeframe: str = "1Min", limit: int = 1000) -> pd.DataFrame:
    """Fetch historical bars from Alpaca Market Data API using the SDK.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., 'AAPL').
    start : str
        ISO date string for the start of the query (UTC).
    end : str
        ISO date string for the end of the query (UTC).
    timeframe : str, optional
        Bar timeframe (e.g., '1Min', '5Min'). Default '1Min'.
    limit : int, optional
        Maximum number of bars to return. Default 1000.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['t','o','h','l','c','v'] representing
        timestamp, open, high, low, close and volume.  Timestamps are
        converted to pandas datetime.
    """
    start_iso = _to_utc_iso(start)
    end_iso = _to_utc_iso(end, default_time="23:59:59")
    barset = client.get_bars(symbol, timeframe, start=start_iso, end=end_iso, limit=limit)
    df = barset.df
    if df.empty:
        raise ValueError("No bar data returned. Check your subscription and date range.")
    # handle multi-index (symbol, timestamp) and single-index cases
    if isinstance(df.index, pd.MultiIndex):
        sym_df = df.xs(symbol, level=0).reset_index()
    else:
        sym_df = df.reset_index()
    sym_df = sym_df.rename(columns={
        "timestamp": "Datetime",
        "time": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    if "Datetime" in sym_df.columns:
        sym_df["Datetime"] = pd.to_datetime(sym_df["Datetime"])
    return sym_df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]


def generate_signals(df: pd.DataFrame, strategy: MeanReversionStrategy) -> pd.DataFrame:
    """Compute signals on fetched bar data."""
    df = df.sort_values("Datetime").reset_index(drop=True)
    signals = strategy.generate_signals(df)
    df["signal"] = signals.values
    return df


def main() -> None:
    cfg = load_config()
    client = get_client(cfg)
    # Fetch data
    df = fetch_bars(
        client,
        cfg["symbol"],
        cfg["start_date"],
        cfg["end_date"],
        cfg.get("timeframe", "1Min"),
    )
    outfile = cfg.get("outfile", "alpaca_data.csv")
    df.to_csv(outfile, index=False)
    print(f"Fetched {len(df)} bars for {cfg['symbol']} from {cfg['start_date']} to {cfg['end_date']}.")
    # Generate signals
    lookback_days = int(cfg.get("lookback_days", 3))
    lookback_bars = lookback_days * 390  # approx minutes per trading day
    strategy = MeanReversionStrategy(
        lookback_bars=lookback_bars,
        upper_pct=float(cfg.get("upper_pct", 0.95)),
        lower_pct=float(cfg.get("lower_pct", 0.05)),
        flat_upper=float(cfg.get("flat_upper", 0.65)),
        flat_lower=float(cfg.get("flat_lower", 0.35)),
    )
    df_with_signals = generate_signals(df, strategy)
    print(df_with_signals[["Datetime", "Close", "signal"]].head(20))
    submit_orders = bool(cfg.get("submit_orders", False))
    if submit_orders:
        # translate signals into position changes; submit bracket orders on transitions
        signals = df_with_signals["signal"].fillna(0).astype(int)
        order_size = int(cfg.get("order_size", 10))
        desired_positions = signals * order_size
        current_position = 0
        last_order_ts: pd.Timestamp | None = None
        min_delta = pd.Timedelta(minutes=float(cfg.get("cooldown_minutes", 5)))
        submitted = []
        for idx, (ts, desired) in enumerate(zip(df_with_signals["Datetime"], desired_positions)):
            diff = desired - current_position
            if diff == 0:
                current_position = desired
                continue
            ts_dt = pd.to_datetime(ts)
            # avoid rapid flip-flops that can trigger wash-trade detection
            if last_order_ts is not None and ts_dt - last_order_ts < min_delta:
                continue
            side = "buy" if diff > 0 else "sell"
            qty = abs(diff)
            price = float(df_with_signals.loc[idx, "Close"])
            tp_pct = float(cfg.get("take_profit_pct", 0.003))
            sl_pct = float(cfg.get("stop_loss_pct", 0.003))
            if side == "buy":
                tp_price = price * (1 + tp_pct)
                sl_price = price * (1 - sl_pct)
            else:
                tp_price = price * (1 - tp_pct)
                sl_price = price * (1 + sl_pct)
            try:
                order = client.submit_order(
                    symbol=cfg["symbol"],
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="day",
                    order_class="bracket",
                    take_profit={"limit_price": tp_price},
                    stop_loss={"stop_price": sl_price},
                )
                submitted.append({"timestamp": ts, "side": side, "qty": qty, "id": getattr(order, "id", None)})
                current_position = desired
                last_order_ts = ts_dt
            except APIError as exc:
                # Skip orders rejected for wash-trade detection or other soft failures
                print(f"Skipped order at {ts} ({side} {qty}): {exc}")
                continue
        print(f"Submitted {len(submitted)} paper orders.")
        for o in submitted[:10]:
            print(o)
        if len(submitted) > 10:
            print("... (truncated)")
    else:
        print("Signal generation only.")


if __name__ == "__main__":
    main()
