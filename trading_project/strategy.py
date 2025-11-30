"""
strategy.py
~~~~~~~~~~~

This module defines a pluggable trading strategy interface and a concrete
implementation of a simple momentum strategy and a mean‑reversion
strategy.  Strategies operate on a cleaned pandas DataFrame containing
historical price data and output a series of trade signals.  The
``Strategy`` base class provides an interface for generating buy/sell
signals, while the ``MomentumStrategy`` computes signals based on the
relationship between short‑ and long‑term moving averages and the
``MeanReversionStrategy`` trades against extremes in rolling returns.

Signals
-------
The strategy returns a pandas Series named ``signal`` indexed by the
DataFrame’s ``Datetime`` index.  Each element takes one of three
values:

``1``   – a buy signal indicating the strategy wishes to be long.

``-1``  – a sell signal indicating the strategy wishes to be short.

``0``   – a neutral signal indicating no position change.

Users are free to extend the base class to implement alternative
strategies.  The momentum strategy here uses a crossover of a fast and
slow moving average as a trend indicator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Abstract base class for trading strategies.

    Subclasses should implement the :meth:`generate_signals` method,
    returning a pandas Series of integer signals aligned with the input
    price data.  A positive signal (1) indicates a long position should
    be taken, negative (-1) indicates a short position, and zero means
    no position.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on input data.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing at least ``Close`` prices with a
            ``Datetime`` index.

        Returns
        -------
        pandas.Series
            Series of integer signals (1, -1, or 0) indexed the same
            as ``data``.
        """
        raise NotImplementedError


@dataclass
class MomentumStrategy(Strategy):
    """Momentum strategy based on moving average crossovers.

    The strategy computes a fast and slow moving average of the closing
    price.  When the fast average crosses above the slow average, a
    bullish momentum is detected and a buy signal (1) is produced.  When
    the fast average crosses below the slow average, a sell signal (-1)
    is produced.  Otherwise the position remains unchanged (0).

    Attributes
    ----------
    fast_window : int
        Length of the fast moving average window in number of bars.

    slow_window : int
        Length of the slow moving average window in number of bars.  Must
        be greater than ``fast_window``.

    threshold : float, optional
        Optional threshold (in price units) to prevent whipsaw trades.
        Signals are only generated when the difference between the fast
        and slow averages exceeds this threshold.  Default is 0.0 (no
        threshold).
    """

    fast_window: int = 20
    slow_window: int = 60
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "Close" not in data.columns:
            raise KeyError("Input data must contain a 'Close' column")
        close = data["Close"].astype(float)
        fast_ma = close.rolling(self.fast_window, min_periods=1).mean()
        slow_ma = close.rolling(self.slow_window, min_periods=1).mean()
        diff = fast_ma - slow_ma
        # Determine signals: 1 for cross above, -1 for cross below
        signals = pd.Series(0, index=data.index, name="signal")
        # compute where diff crosses threshold
        prev_diff = diff.shift(1)
        # buy when diff crosses above threshold
        buy_mask = (diff > self.threshold) & (prev_diff <= self.threshold)
        # sell when diff crosses below -threshold
        sell_mask = (diff < -self.threshold) & (prev_diff >= -self.threshold)
        signals.loc[buy_mask] = 1
        signals.loc[sell_mask] = -1
        return signals


@dataclass
class MeanReversionStrategy(Strategy):
    """Mean-reversion strategy based on rolling return percentiles.

    The strategy looks at a rolling window of returns and enters against
    extremes: it shorts when the current return exceeds the upper
    percentile (e.g., 95th) and buys when it is below the lower
    percentile (e.g., 5th). Positions are closed when the return mean
    reverts back inside a central percentile band (e.g., 65th/35th).

    Attributes
    ----------
    lookback_bars : int
        Number of bars to look back for percentile calculations (e.g.,
        ~390 bars per trading day for 1m data).
    upper_pct : float
        Upper percentile threshold to trigger a short (default 0.95).
    lower_pct : float
        Lower percentile threshold to trigger a long (default 0.05).
    flat_upper : float
        Upper percentile threshold to exit a short (default 0.65).
    flat_lower : float
        Lower percentile threshold to exit a long (default 0.35).
    """

    lookback_bars: int = 390 * 3  # ~3 trading days of 1m bars
    upper_pct: float = 0.95
    lower_pct: float = 0.05
    flat_upper: float = 0.65
    flat_lower: float = 0.35

    def __post_init__(self) -> None:
        if not (0 < self.lower_pct < self.upper_pct < 1):
            raise ValueError("Percentiles must satisfy 0 < lower_pct < upper_pct < 1")
        if not (0 < self.flat_lower < self.flat_upper < 1):
            raise ValueError("Flat band percentiles must satisfy 0 < flat_lower < flat_upper < 1")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "Close" not in data.columns:
            raise KeyError("Input data must contain a 'Close' column")
        close = data["Close"].astype(float)
        returns = close.pct_change()
        roll = returns.rolling(self.lookback_bars, min_periods=max(10, self.lookback_bars // 2))
        upper = roll.quantile(self.upper_pct)
        lower = roll.quantile(self.lower_pct)
        flat_upper = roll.quantile(self.flat_upper)
        flat_lower = roll.quantile(self.flat_lower)

        signals = pd.Series(0, index=data.index, name="signal")
        position = 0
        for idx in range(len(data)):
            r = returns.iloc[idx]
            u = upper.iloc[idx]
            l = lower.iloc[idx]
            fu = flat_upper.iloc[idx]
            fl = flat_lower.iloc[idx]
            if np.isnan(r) or np.isnan(u) or np.isnan(l) or np.isnan(fu) or np.isnan(fl):
                signals.iloc[idx] = position
                continue
            if position == 0:
                if r > u:
                    position = -1
                elif r < l:
                    position = 1
            elif position == 1 and r >= fl:
                position = 0
            elif position == -1 and r <= fu:
                position = 0
            signals.iloc[idx] = position
        return signals


def build_strategy_from_config(cfg: Dict[str, Any]) -> Strategy:
    """Factory to construct a Strategy from config settings."""
    strategy_type = cfg.get("strategy", "mean_reversion")
    if strategy_type != "mean_reversion":
        raise ValueError(f"Unsupported strategy type: {strategy_type}")
    lookback_bars = cfg.get("lookback_bars")
    if lookback_bars is None:
        lookback_days = int(cfg.get("lookback_days", 3))
        lookback_bars = lookback_days * 390
    return MeanReversionStrategy(
        lookback_bars=int(lookback_bars),
        upper_pct=float(cfg.get("upper_pct", 0.95)),
        lower_pct=float(cfg.get("lower_pct", 0.05)),
        flat_upper=float(cfg.get("flat_upper", 0.65)),
        flat_lower=float(cfg.get("flat_lower", 0.35)),
    )
