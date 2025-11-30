"""
alpaca_live_check.py
~~~~~~~~~~~~~~~~~~~~

Safety check script to verify paper-trading connectivity without touching
your main strategy flow. It reads credentials/settings from
``alpaca_config.json`` and performs:

1) Account/clock check (paper only)
2) Recent bar fetch (last day) to confirm market data access
3) Optional tiny test order (disabled by default) to verify trading
   permissions; if enabled, it submits and immediately cancels a 1-share
   market order.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

CONFIG_PATH = Path(__file__).resolve().parent / "alpaca_config.json"

# Set to True to place and immediately cancel a tiny test order (paper).
PLACE_TEST_ORDER = False
TEST_SYMBOL = "AAPL"
TEST_QTY = 1


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("api_key", "api_secret", "base_url"):
        if not cfg.get(key):
            raise ValueError(f"Config missing required key: {key}")
    return cfg


def get_client(cfg: Dict[str, Any]) -> tradeapi.REST:
    return tradeapi.REST(cfg["api_key"], cfg["api_secret"], cfg["base_url"], api_version="v2")


def check_account(api: tradeapi.REST) -> None:
    acct = api.get_account()
    print(f"Account status: {acct.status}, equity: {acct.equity}, buying_power: {acct.buying_power}")
    clock = api.get_clock()
    print(f"Market is_open={clock.is_open}, next_open={clock.next_open}, next_close={clock.next_close}")


def check_bars(api: tradeapi.REST, symbol: str) -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=1)
    try:
        bars = api.get_bars(symbol, "1Min", start=start.isoformat(), end=end.isoformat(), limit=5, feed="iex").df
    except APIError as exc:
        print(f"Bar fetch failed (possible data plan issue): {exc}")
        return
    if bars.empty:
        print(f"No bars returned for {symbol} in the last day (feed=iex).")
    else:
        print(f"Recent bars for {symbol} (feed=iex):")
        print(bars.tail(3))


def test_order(api: tradeapi.REST) -> None:
    try:
        order = api.submit_order(
            symbol=TEST_SYMBOL,
            qty=TEST_QTY,
            side="buy",
            type="market",
            time_in_force="day",
        )
        print(f"Placed test order id={order.id}")
        try:
            api.cancel_order(order.id)
            print("Cancelled test order.")
        except APIError as exc:
            print(f"Could not cancel test order: {exc}")
    except APIError as exc:
        print(f"Test order failed: {exc}")


def main() -> None:
    cfg = load_config()
    api = get_client(cfg)
    check_account(api)
    check_bars(api, cfg.get("symbol", TEST_SYMBOL))
    if PLACE_TEST_ORDER:
        test_order(api)
    else:
        print("Test order disabled (set PLACE_TEST_ORDER=True to try).")


if __name__ == "__main__":
    main()
