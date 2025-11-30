"""
test_live_runner.py
~~~~~~~~~~~~~~~~~~~

Single-pass sanity check for the live runner components without
submitting any orders. It will:
  - Load config
  - Build the Alpaca client
  - Check account/clock
  - Fetch recent bars (last day, feed=iex)
  - Run the configured strategy to produce the latest signal
  - Report desired vs current position and equity snapshot

No orders are submitted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

from alpaca_integration import load_config, get_client, _to_utc_iso
from strategy import build_strategy_from_config
from live_runner import current_position_qty


def fetch_recent_bars(api: tradeapi.REST, symbol: str) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=1)
    try:
        bars = api.get_bars(
            symbol,
            "1Min",
            start=_to_utc_iso(start.isoformat()),
            end=_to_utc_iso(end.isoformat(), default_time="23:59:59"),
            feed="iex",
            limit=2000,
        ).df
    except APIError as exc:
        raise RuntimeError(f"Bar fetch failed: {exc}") from exc
    if bars.empty:
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        sym_df = bars.xs(symbol, level=0).reset_index()
    else:
        sym_df = bars.reset_index()
    sym_df = sym_df.rename(columns={
        "timestamp": "Datetime",
        "time": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    sym_df["Datetime"] = pd.to_datetime(sym_df["Datetime"])
    return sym_df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].sort_values("Datetime")


def main() -> None:
    cfg = load_config()
    api = get_client(cfg)
    symbol = cfg["symbol"]

    acct = api.get_account()
    clock = api.get_clock()
    print(f"Account status={acct.status}, equity={acct.equity}, buying_power={acct.buying_power}")
    print(f"Market is_open={clock.is_open}, next_open={clock.next_open}, next_close={clock.next_close}")

    df = fetch_recent_bars(api, symbol)
    if df.empty:
        print(f"No bars returned for {symbol} (last day, feed=iex).")
        return

    strategy = build_strategy_from_config(cfg)
    signals = strategy.generate_signals(df)
    latest_signal = int(signals.iloc[-1])
    order_size = int(cfg.get("order_size", 10))
    desired_position = latest_signal * order_size
    current_position = current_position_qty(api, symbol)

    print(f"Fetched {len(df)} bars; latest close={df.iloc[-1]['Close']}")
    print(f"Latest signal={latest_signal}, desired_position={desired_position}, current_position={current_position}")
    print("No orders submitted in this test run.")


if __name__ == "__main__":
    main()
