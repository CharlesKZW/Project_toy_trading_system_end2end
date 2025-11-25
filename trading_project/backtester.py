"""
backtester.py
~~~~~~~~~~~~~

This module implements a minimalistic backtesting framework.  It simulates a
simple order lifecycle through a gateway, order manager, order book and
matching engine to approximate the behaviour of a trading venue.  The
framework does not attempt to replicate all of the intricacies of a real
exchange; rather it provides a sandbox for testing algorithmic trading
strategies on historical intraday data.

Major components
----------------

Gateway
    Reads historical market data from a CSV file or DataFrame and yields
    successive rows to simulate a live feed.  The gateway can also log
    orders sent, modified, cancelled or filled to a file for later audit.

OrderBook
    Stores outstanding bid and ask orders using priority queues.  Buy
    orders are prioritised by highest price then earliest time, while
    sell orders are prioritised by lowest price then earliest time.  The
    class supports adding orders and retrieving the best available order
    for matching.

OrderManager
    Validates orders against capital constraints and risk limits before
    submitting them to the order book.  It maintains cash and position
    information and enforces limits such as maximum orders per minute and
    maximum position size.

MatchingEngine
    Consumes orders from the order book and determines execution
    outcomes.  For simplicity this implementation randomly decides
    whether orders are filled, partially filled or cancelled.  Filled
    quantities and prices are returned for downstream accounting.

Backtester
    Integrates the above components with a trading strategy.  On each
    bar of market data the strategy generates a signal, orders are
    created accordingly, validated by the manager, executed by the
    matching engine, and positions and cash are updated.  After all
    bars are processed performance metrics such as total P&L, Sharpe
    ratio, maximum drawdown and win/loss ratio are computed.

Limitations
-----------
This framework is intentionally simplified.  Execution latency,
transaction costs, slippage and realistic matching logic are not
modelled.  Users should exercise caution when extrapolating backtest
results to real markets.
"""

from __future__ import annotations

import csv
import datetime as dt
import heapq
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Order:
    """Represents a trade order."""

    order_id: int
    timestamp: pd.Timestamp
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    price: float


class Gateway:
    """Simulate a live data feed by iterating over rows of a DataFrame.

    The gateway yields each row one by one.  It also records order events to
    a log file on disk for post‑trade analysis.
    """

    def __init__(self, data: pd.DataFrame, log_path: str = "order_log.csv") -> None:
        self.data = data.reset_index(drop=True)
        self.log_path = log_path
        # Prepare log file
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "order_id",
                "event",
                "side",
                "quantity",
                "price",
                "status",
                "filled_qty",
                "filled_price",
            ])

    def stream(self) -> Iterator[pd.Series]:
        """Yield one row of market data at a time."""
        for _, row in self.data.iterrows():
            yield row

    def log_order(self, order: Order, event: str, status: str,
                  filled_qty: int = 0, filled_price: float = 0.0) -> None:
        """Record an order event to the log file.

        Parameters
        ----------
        order : Order
            The order being logged.
        event : str
            Event type ('submitted', 'validated', 'rejected', 'filled', 'partial',
            'cancelled').
        status : str
            Current status of the order.
        filled_qty : int, optional
            Quantity filled in this event.
        filled_price : float, optional
            Price at which the order was filled in this event.
        """
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                order.timestamp,
                order.order_id,
                event,
                order.side,
                order.quantity,
                order.price,
                status,
                filled_qty,
                filled_price,
            ])


class OrderBook:
    """Maintain priority queues for buy and sell orders."""

    def __init__(self) -> None:
        # max-heap for buys: invert price to use heapq (min-heap)
        self.buys: List[Tuple[float, pd.Timestamp, Order]] = []
        # min-heap for sells
        self.sells: List[Tuple[float, pd.Timestamp, Order]] = []

    def add_order(self, order: Order) -> None:
        if order.side == "buy":
            heapq.heappush(self.buys, (-order.price, order.timestamp, order))
        else:
            heapq.heappush(self.sells, (order.price, order.timestamp, order))

    def pop_best_buy(self) -> Optional[Order]:
        if not self.buys:
            return None
        _, _, order = heapq.heappop(self.buys)
        return order

    def pop_best_sell(self) -> Optional[Order]:
        if not self.sells:
            return None
        _, _, order = heapq.heappop(self.sells)
        return order


class OrderManager:
    """Validate and manage orders before execution."""

    def __init__(self, initial_capital: float = 100_000.0,
                 max_position: int = 1_000, max_orders_per_minute: int = 10) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0  # positive for long, negative for short
        self.max_position = max_position
        self.max_orders_per_minute = max_orders_per_minute
        self.order_count_in_minute: Dict[pd.Timestamp, int] = {}

    def validate(self, order: Order, current_price: float) -> bool:
        """Check capital and risk limits for the order.

        Returns True if the order is accepted, False otherwise.
        """
        # limit on number of orders per minute
        ts_minute = order.timestamp.floor("T")
        count = self.order_count_in_minute.get(ts_minute, 0)
        if count >= self.max_orders_per_minute:
            return False
        # check capital for buy orders
        if order.side == "buy":
            cost = order.price * order.quantity
            if self.cash < cost:
                return False
            if self.position + order.quantity > self.max_position:
                return False
        else:  # sell order
            # ensure we have enough position to short beyond negative max_position
            if self.position - order.quantity < -self.max_position:
                return False
        return True

    def record_order(self, order: Order) -> None:
        # update order count for minute
        ts_minute = order.timestamp.floor("T")
        self.order_count_in_minute[ts_minute] = self.order_count_in_minute.get(ts_minute, 0) + 1

    def update_after_fill(self, order: Order, fill_qty: int, fill_price: float) -> None:
        """Adjust cash and position after a fill or partial fill."""
        if fill_qty == 0:
            return
        if order.side == "buy":
            self.position += fill_qty
            self.cash -= fill_price * fill_qty
        else:
            self.position -= fill_qty
            self.cash += fill_price * fill_qty

    def current_value(self, current_price: float) -> float:
        """Compute total account value given the current market price."""
        return self.cash + self.position * current_price


class MatchingEngine:
    """Simulate order matching and execution outcomes."""

    def __init__(self, fill_probability: float = 0.7) -> None:
        # probability that an order will be filled rather than cancelled
        self.fill_probability = fill_probability

    def match(self, order: Order, current_price: float) -> Tuple[int, float, str]:
        """Determine the execution outcome for an order.

        Returns a tuple of (filled_quantity, fill_price, status), where status
        is one of 'filled', 'partial', or 'cancelled'.
        """
        rnd = random.random()
        if rnd > self.fill_probability:
            return 0, 0.0, "cancelled"
        # determine fill quantity: fully fill 70% of accepted orders, partial otherwise
        if random.random() < 0.7:
            fill_qty = order.quantity
            status = "filled"
        else:
            # partial fill between 10% and 90%
            fill_qty = max(1, int(order.quantity * random.uniform(0.1, 0.9)))
            status = "partial"
        # execution price: assume price improvement within spread
        price_variation = random.uniform(-0.01, 0.01)  # ±1 cent variation
        fill_price = max(0.01, order.price + price_variation)
        return fill_qty, fill_price, status


class Backtester:
    """Backtest a trading strategy on historical data."""

    def __init__(self, strategy: Callable[[pd.DataFrame], pd.Series],
                 data: pd.DataFrame, order_size: int = 10,
                 initial_capital: float = 100_000.0) -> None:
        self.strategy = strategy
        self.data = data.copy().reset_index(drop=True)
        self.order_size = order_size
        self.gateway = Gateway(self.data, log_path="order_log.csv")
        self.order_book = OrderBook()
        self.manager = OrderManager(initial_capital=initial_capital)
        self.engine = MatchingEngine(fill_probability=0.8)
        self.orders: Dict[int, Order] = {}
        self.next_order_id = 1
        # results
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.trades: List[Dict[str, float]] = []

    def run(self) -> None:
        # Generate signals for the entire dataset up front
        signals = self.strategy.generate_signals(self.data)
        current_position = 0
        for idx, row in self.data.iterrows():
            timestamp = pd.to_datetime(row["Datetime"])
            price = float(row["Close"])
            signal = signals.iloc[idx] if idx < len(signals) else 0
            # Determine desired position based on signal
            # 1 -> long, -1 -> short, 0 -> flat
            desired_position = signal * self.order_size
            if desired_position != 0 and desired_position != current_position:
                # compute order side and quantity difference
                diff = desired_position - current_position
                side = "buy" if diff > 0 else "sell"
                quantity = abs(diff)
                order = Order(
                    order_id=self.next_order_id,
                    timestamp=timestamp,
                    symbol="AAPL",
                    side=side,
                    quantity=quantity,
                    price=price,
                )
                self.next_order_id += 1
                # log submitted
                self.gateway.log_order(order, event="submitted", status="new")
                # validate
                if self.manager.validate(order, price):
                    self.gateway.log_order(order, event="validated", status="accepted")
                    self.manager.record_order(order)
                    # matching
                    fill_qty, fill_price, status = self.engine.match(order, price)
                    if status == "cancelled":
                        self.gateway.log_order(order, event="cancelled", status=status)
                    else:
                        self.manager.update_after_fill(order, fill_qty, fill_price)
                        self.gateway.log_order(order, event="filled" if status == "filled" else "partial",
                                              status=status, filled_qty=fill_qty,
                                              filled_price=fill_price)
                        # record trade
                        self.trades.append({
                            "timestamp": timestamp,
                            "side": side,
                            "quantity": fill_qty,
                            "price": fill_price,
                        })
                        # update current position based on fill
                        current_position += fill_qty if side == "buy" else -fill_qty
                else:
                    self.gateway.log_order(order, event="rejected", status="rejected")
            # update equity curve
            account_value = self.manager.current_value(price)
            self.equity_curve.append((timestamp, account_value))

    def results(self) -> Dict[str, float]:
        """Compute performance metrics from the equity curve."""
        if not self.equity_curve:
            return {}
        # Build pandas Series for equity curve
        index = [ts for ts, _ in self.equity_curve]
        equity = pd.Series([v for _, v in self.equity_curve], index=pd.to_datetime(index))
        # Compute returns
        returns = equity.pct_change().dropna()
        # Total P&L
        total_pnl = equity.iloc[-1] - self.manager.initial_capital
        # Sharpe ratio: assume 252 trading days, 390 minutes/day -> 252*390 periods
        # convert to daily returns: multiply minute returns by sqrt(periods per day)
        periods_per_day = 390
        if len(returns) > 0:
            mean_return = returns.mean() * periods_per_day
            std_return = returns.std() * np.sqrt(periods_per_day)
            sharpe = (mean_return / std_return) if std_return != 0 else 0.0
        else:
            sharpe = 0.0
        # drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        # win/loss ratio based on trades
        wins = 0
        losses = 0
        for trade in self.trades:
            # approximate profit per trade: position * price change over next bar
            # For simplicity evaluate profit at next bar close
            ts = trade["timestamp"]
            side = trade["side"]
            qty = trade["quantity"]
            entry_price = trade["price"]
            # find next bar
            idx = equity.index.get_indexer([ts], method="nearest")[0]
            if idx + 1 < len(equity):
                exit_price = self.data.loc[idx + 1, "Close"]
                pnl = (exit_price - entry_price) * qty if side == "buy" else (entry_price - exit_price) * qty
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
        win_loss_ratio = (wins / losses) if losses > 0 else float('inf') if wins > 0 else 0.0
        return {
            "total_pnl": total_pnl,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_loss_ratio": win_loss_ratio,
            "final_value": equity.iloc[-1],
            "num_trades": len(self.trades),
        }
