from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    """Metadata + callable for one strategy."""

    name: str
    description: str
    fn: Callable[[pd.DataFrame], pd.Series]


_DAYS_PER_MONTH = 21
_DAYS_PER_YEAR = 252


def _ts_momentum_12_1(df: pd.DataFrame) -> pd.Series:
    """Time-series momentum: sign of 12-month return, skipping the last month.

    Skipping the most recent month is a classic adjustment from Jegadeesh &
    Titman (1993) to mitigate short-term reversal effects.
    """
    close = df["Close"]
    # Return from t-12m to t-1m, computed at day t.
    long_ago = close.shift(_DAYS_PER_YEAR)
    one_month_ago = close.shift(_DAYS_PER_MONTH)
    mom = one_month_ago / long_ago - 1
    return (mom > 0).astype(int).fillna(0).astype(int)


def _ma_crossover_50_200(df: pd.DataFrame) -> pd.Series:
    """Golden cross: long when 50-day SMA > 200-day SMA, flat otherwise."""
    close = df["Close"]
    ma_fast = close.rolling(50).mean()
    ma_slow = close.rolling(200).mean()
    return (ma_fast > ma_slow).astype(int).fillna(0).astype(int)


def _dual_momentum(df: pd.DataFrame) -> pd.Series:
    """Long if 12-month return is positive, else flat.

    Simplified single-asset version of Antonacci's dual momentum: in a
    multi-asset universe, you'd also rank assets and pick the best. Here we
    only have SPY, so we just check absolute momentum vs cash.
    """
    close = df["Close"]
    ret_12m = close.pct_change(_DAYS_PER_YEAR)
    return (ret_12m > 0).astype(int).fillna(0).astype(int)


def _donchian_breakout_20(df: pd.DataFrame) -> pd.Series:
    """Long when close breaks above 20-day high; exit below 20-day low.

    The 20-day high/low are computed using only past data (shift by 1) to
    avoid using today's close as part of today's signal.
    """
    close = df["Close"]
    lookback = 20
    upper = close.rolling(lookback).max().shift(1)
    lower = close.rolling(lookback).min().shift(1)

    # Stateful: once long, stay long until a downside break.
    state = np.zeros(len(close), dtype=int)
    current = 0
    close_vals = close.to_numpy()
    upper_vals = upper.to_numpy()
    lower_vals = lower.to_numpy()

    for i in range(len(close_vals)):
        if np.isnan(upper_vals[i]) or np.isnan(lower_vals[i]):
            state[i] = 0
            continue
        if current == 0 and close_vals[i] > upper_vals[i]:
            current = 1
        elif current == 1 and close_vals[i] < lower_vals[i]:
            current = 0
        state[i] = current

    return pd.Series(state, index=close.index, dtype=int)


def _risk_adjusted_momentum(df: pd.DataFrame) -> pd.Series:
    """Long when (6-month return / 6-month annualized vol) is positive.

    Equivalent to a momentum signal scaled by vol — a poor man's Sharpe filter.
    """
    close = df["Close"]
    lookback = _DAYS_PER_MONTH * 6  # ~126 trading days
    rets = close.pct_change()
    mom_6m = close.pct_change(lookback)
    vol_6m = rets.rolling(lookback).std() * np.sqrt(_DAYS_PER_YEAR)
    score = mom_6m / vol_6m
    return (score > 0).astype(int).fillna(0).astype(int)


STRATEGIES: dict[str, StrategySpec] = {
    "ts_momentum_12_1": StrategySpec(
        name="ts_momentum_12_1",
        description=(
            "Time-series momentum: long when the 12-month return (skipping the "
            "most recent month) is positive, flat otherwise."
        ),
        fn=_ts_momentum_12_1,
    ),
    "ma_crossover_50_200": StrategySpec(
        name="ma_crossover_50_200",
        description=(
            "Golden cross: long when the 50-day SMA is above the 200-day SMA, flat otherwise."
        ),
        fn=_ma_crossover_50_200,
    ),
    "dual_momentum": StrategySpec(
        name="dual_momentum",
        description=(
            "Long when the 12-month return is positive, flat otherwise. "
            "Single-asset variant of Antonacci's dual momentum."
        ),
        fn=_dual_momentum,
    ),
    "donchian_breakout_20": StrategySpec(
        name="donchian_breakout_20",
        description=(
            "Long on a breakout above the 20-day high; exit on a break below the 20-day low."
        ),
        fn=_donchian_breakout_20,
    ),
    "risk_adjusted_momentum": StrategySpec(
        name="risk_adjusted_momentum",
        description=(
            "Long when the 6-month return divided by 6-month annualized volatility is positive."
        ),
        fn=_risk_adjusted_momentum,
    ),
}
