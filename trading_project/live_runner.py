"""
live_runner.py
~~~~~~~~~~~~~~

Minute-by-minute paper trading loop:
  - Loads config from alpaca_config.json
  - Waits for market open (paper clock)
  - Each minute: fetches new bars, appends to a local CSV, runs the
    mean-reversion strategy, reconciles with current position, and
    submits bracket orders if needed. Logs orders and account snapshots.

This script uses the IEX feed to avoid SIP data restrictions on many
paper plans. It is paper-only; ensure your config points to the paper
endpoint.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

from alpaca_integration import (
    load_config,
    get_client,
    generate_signals,
    _to_utc_iso,
)
from strategy import build_strategy_from_config


def load_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df = pd.read_csv(path, parse_dates=["Datetime"])
    return df


def append_bars(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new.empty:
        return existing
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    return combined


def fetch_incremental_bars(api: tradeapi.REST, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    start_iso = _to_utc_iso(start_ts.isoformat())
    end_iso = _to_utc_iso(end_ts.isoformat(), default_time="23:59:59")
    bars = api.get_bars(symbol, "1Min", start=start_iso, end=end_iso, feed="iex").df
    if bars.empty:
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
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
    return sym_df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]


def current_position_qty(api: tradeapi.REST, symbol: str) -> int:
    try:
        positions = api.list_positions()
    except APIError:
        return 0
    qty = 0
    for pos in positions:
        if pos.symbol == symbol:
            try:
                qty += int(float(pos.qty))
            except ValueError:
                continue
    return qty


def submit_bracket(api: tradeapi.REST, symbol: str, side: str, qty: int, price: float,
                   tp_pct: float, sl_pct: float) -> Dict[str, Any]:
    if side == "buy":
        tp_price = price * (1 + tp_pct)
        sl_price = price * (1 - sl_pct)
    else:
        tp_price = price * (1 - tp_pct)
        sl_price = price * (1 + sl_pct)
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day",
        order_class="bracket",
        take_profit={"limit_price": tp_price},
        stop_loss={"stop_price": sl_price},
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "tp": tp_price,
        "sl": sl_price,
        "id": getattr(order, "id", None),
    }


def log_account(api: tradeapi.REST, symbol: str) -> Dict[str, Any]:
    acct = api.get_account()
    pos_qty = current_position_qty(api, symbol)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "equity": acct.equity,
        "cash": acct.cash,
        "buying_power": acct.buying_power,
        "position_qty": pos_qty,
    }


def sleep_to_next_minute():
    now = datetime.now()
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    delta = (next_minute - now).total_seconds()
    time.sleep(max(1, delta))


def main() -> None:
    cfg = load_config()
    api = get_client(cfg)
    symbol = cfg["symbol"]
    data_path = Path(cfg.get("outfile", "alpaca_data.csv"))
    order_log_path = Path(cfg.get("submitted_log", "alpaca_submitted_orders.csv"))
    equity_log_path = Path(cfg.get("equity_log", "alpaca_equity_log.csv"))
    order_size = int(cfg.get("order_size", 10))
    tp_pct = float(cfg.get("take_profit_pct", 0.003))
    sl_pct = float(cfg.get("stop_loss_pct", 0.003))
    cooldown_min = float(cfg.get("cooldown_minutes", 5))
    min_delta = pd.Timedelta(minutes=cooldown_min)
    lookback_days = int(cfg.get("lookback_days", 3))

    # Strategy
    strategy = build_strategy_from_config(cfg)

    # Wait for market open
    while True:
        clock = api.get_clock()
        if clock.is_open:
            break
        wait_seconds = max(60, (clock.next_open - datetime.now(timezone.utc)).total_seconds())
        print(f"Market closed. Sleeping {int(wait_seconds)}s until next open.")
        time.sleep(min(wait_seconds, 300))

    data = load_bars(data_path)
    last_order_ts = None
    while True:
        clock = api.get_clock()
        if not clock.is_open:
            print("Market closed. Exiting loop.")
            break

        now = pd.Timestamp.utcnow()
        if data.empty:
            start_ts = now - pd.Timedelta(days=lookback_days)
        else:
            start_ts = data["Datetime"].max() + pd.Timedelta(minutes=1)

        try:
            new_bars = fetch_incremental_bars(api, symbol, start_ts, now)
        except APIError as exc:
            print(f"Bar fetch failed: {exc}")
            sleep_to_next_minute()
            continue

        data = append_bars(data, new_bars)
        data.to_csv(data_path, index=False)

        if data.empty:
            print("No data available yet; sleeping to next minute.")
            sleep_to_next_minute()
            continue

        df_with_signals = generate_signals(data, strategy)
        signals = df_with_signals["signal"].fillna(0).astype(int)
        desired_position = signals.iloc[-1] * order_size
        current_position = current_position_qty(api, symbol)

        if desired_position != current_position:
            diff = desired_position - current_position
            side = "buy" if diff > 0 else "sell"
            qty = abs(diff)
            ts_dt = pd.to_datetime(df_with_signals.iloc[-1]["Datetime"])
            if last_order_ts is None or ts_dt - last_order_ts >= min_delta:
                price = float(df_with_signals.iloc[-1]["Close"])
                try:
                    submitted = submit_bracket(api, symbol, side, qty, price, tp_pct, sl_pct)
                    last_order_ts = ts_dt
                    pd.DataFrame([submitted]).to_csv(order_log_path, mode="a", header=not order_log_path.exists(), index=False)
                    print(f"Submitted order: {submitted}")
                except APIError as exc:
                    print(f"Order submission failed: {exc}")
            else:
                print("Cooldown active; skipping order this minute.")

        snapshot = log_account(api, symbol)
        pd.DataFrame([snapshot]).to_csv(equity_log_path, mode="a", header=not equity_log_path.exists(), index=False)

        sleep_to_next_minute()


if __name__ == "__main__":
    main()
