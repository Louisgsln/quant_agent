from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_agent.backtest.engine import run_backtest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_prices() -> pd.DataFrame:
    """100 business days of constant price = no return possible."""
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    return pd.DataFrame({"Close": np.full(100, 100.0)}, index=idx)


@pytest.fixture
def trending_up() -> pd.DataFrame:
    """100 business days, ~0.1%/day uptrend."""
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    prices = 100 * (1 + 0.001) ** np.arange(100)
    return pd.DataFrame({"Close": prices}, index=idx)


@pytest.fixture
def alternating_prices() -> pd.DataFrame:
    """5-day cycle: 100, 110, 100, 110, 100, ... — useful for lookahead tests."""
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    return pd.DataFrame({"Close": [100, 110, 100, 110, 100, 110, 100, 110, 100, 110]}, index=idx)


# ---------------------------------------------------------------------------
# Contract / sanity
# ---------------------------------------------------------------------------


def test_returns_aligned_on_input_index(flat_prices: pd.DataFrame) -> None:
    """All output series must share the input's DatetimeIndex."""
    pos = pd.Series(1, index=flat_prices.index)
    result = run_backtest(flat_prices, pos)
    assert result.returns.index.equals(flat_prices.index)
    assert result.positions.index.equals(flat_prices.index)
    assert result.equity.index.equals(flat_prices.index)
    assert result.benchmark_returns.index.equals(flat_prices.index)


def test_validates_missing_close_column(flat_prices: pd.DataFrame) -> None:
    """Missing 'Close' column must raise a clear error."""
    bad = flat_prices.rename(columns={"Close": "close"})
    pos = pd.Series(1, index=flat_prices.index)
    with pytest.raises(ValueError, match="Close"):
        run_backtest(bad, pos)


def test_validates_index_mismatch(flat_prices: pd.DataFrame) -> None:
    """Prices and positions must share the same index."""
    other_idx = pd.date_range("2024-01-01", periods=100, freq="B")
    pos = pd.Series(1, index=other_idx)
    with pytest.raises(ValueError, match="same index"):
        run_backtest(flat_prices, pos)


# ---------------------------------------------------------------------------
# Behavioral
# ---------------------------------------------------------------------------


def test_zero_position_zero_return(flat_prices: pd.DataFrame) -> None:
    """Always-flat strategy returns nothing on flat prices."""
    pos = pd.Series(0, index=flat_prices.index)
    result = run_backtest(flat_prices, pos)
    assert result.returns.sum() == 0
    assert result.equity.iloc[-1] == 1.0


def test_full_long_matches_benchmark_when_no_costs(trending_up: pd.DataFrame) -> None:
    """A strategy always long should match buy-and-hold exactly when costs=0."""
    pos = pd.Series(1, index=trending_up.index)
    result = run_backtest(trending_up, pos, cost_bps=0)

    # Note: due to the shift(1), day 0 has position 0, so the strategy misses
    # the first day's return. Compare from day 1 onwards.
    pd.testing.assert_series_equal(
        result.returns.iloc[1:],
        result.benchmark_returns.iloc[1:],
        check_names=False,
    )


# ---------------------------------------------------------------------------
# THE critical test
# ---------------------------------------------------------------------------


def test_no_lookahead_bias(alternating_prices: pd.DataFrame) -> None:
    """Position decided at close of day t must apply to return of day t+1.

    Setup: prices alternate 100/110 every day. Returns are +10% / -9.09%.
    A 'perfect foresight' position series would be [1, 0, 1, 0, ...] (long on
    days where the next return is positive). With proper shift(1):
    - Day 0: applied_pos = 0 (shifted from NaN)
    - Day 1: applied_pos = 1 (shifted from day 0's signal) → captures +10%
    - Day 2: applied_pos = 0 (shifted from day 1) → misses the -9.09%
    - ...
    With WRONG implementation (no shift), day 0's pos=1 would apply to day 0's
    return (which is 0 because it's pct_change of the first row). The first
    real return captured would be day 1's +10% but that's coincidentally the
    same. The test makes both implementations distinguishable on day 2:
    - Correct: day 2 captures -9.09% × 0 = 0
    - Wrong:   day 2 captures -9.09% × 1 = -9.09%
    """
    # "Perfect" signal: long if NEXT day's return is positive.
    # Returns: [NaN, +10%, -9.09%, +10%, -9.09%, ...], so signal at t looks at t+1.
    # We hardcode the signal that a clairvoyant trader would use at close of t.
    pos = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], index=alternating_prices.index)
    result = run_backtest(alternating_prices, pos, cost_bps=0)

    # Day 1: applied_pos = 1 (from day 0's signal), return = +10% → captured.
    assert abs(result.returns.iloc[1] - 0.10) < 1e-9

    # Day 2: applied_pos = 0 (from day 1's signal), return = -9.09% → missed.
    assert result.returns.iloc[2] == 0.0

    # If lookahead were present, returns.iloc[2] would be -0.0909 (the wrong path).


def test_costs_reduce_returns_proportionally(flat_prices: pd.DataFrame) -> None:
    """Higher cost_bps should produce strictly lower returns on a turnover-heavy strategy."""
    # Position flips every day → high turnover.
    pos = pd.Series([0, 1] * 50, index=flat_prices.index)

    res_no_cost = run_backtest(flat_prices, pos, cost_bps=0)
    res_low = run_backtest(flat_prices, pos, cost_bps=5)
    res_high = run_backtest(flat_prices, pos, cost_bps=20)

    assert res_no_cost.returns.sum() == 0  # flat prices, no cost → break even
    assert res_low.returns.sum() < 0
    assert res_high.returns.sum() < res_low.returns.sum()


def test_buy_and_hold_has_zero_turnover(trending_up: pd.DataFrame) -> None:
    """Permanently long: only one position change at the very start (0 → 1 via shift)."""
    pos = pd.Series(1, index=trending_up.index)
    result = run_backtest(trending_up, pos, cost_bps=10)

    # The only turnover comes from the initial shift(1) day-0 transition.
    # Total turnover should be exactly 1 (one unit of position taken).
    total_turnover = result.positions.diff().abs().fillna(0).sum() + result.positions.iloc[0]
    assert abs(total_turnover - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Integration: realistic case
# ---------------------------------------------------------------------------


def test_summary_returns_expected_keys(trending_up: pd.DataFrame) -> None:
    """The .summary() helper exposes useful aggregates."""
    pos = pd.Series([1] * 50 + [0] * 50, index=trending_up.index)
    result = run_backtest(trending_up, pos)

    summary = result.summary()
    assert set(summary.keys()) == {"n_days", "final_equity", "benchmark_final_equity", "turnover_total"}
    assert summary["n_days"] == 100
    assert summary["final_equity"] > 0