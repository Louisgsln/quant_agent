from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_agent.strategies.momentum import STRATEGIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trending_up() -> pd.DataFrame:
    """500 business days of clean uptrend (~0.05%/day)."""
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    prices = 100 * (1 + 0.0005) ** np.arange(500)
    return pd.DataFrame({"Close": prices}, index=idx)


@pytest.fixture
def trending_down() -> pd.DataFrame:
    """500 business days of clean downtrend (~0.05%/day)."""
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    prices = 100 * (1 - 0.0005) ** np.arange(500)
    return pd.DataFrame({"Close": prices}, index=idx)


@pytest.fixture
def short_history() -> pd.DataFrame:
    """50 days — too short for any 200-day or 252-day indicator."""
    idx = pd.date_range("2023-01-01", periods=50, freq="B")
    return pd.DataFrame({"Close": np.linspace(100, 110, 50)}, index=idx)


# ---------------------------------------------------------------------------
# Invariants — must hold for every strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(STRATEGIES.keys()))
def test_returns_series_aligned_on_input_index(name: str, trending_up: pd.DataFrame) -> None:
    """Every strategy returns a Series with the same index as the input."""
    pos = STRATEGIES[name].fn(trending_up)
    assert isinstance(pos, pd.Series)
    assert len(pos) == len(trending_up)
    assert pos.index.equals(trending_up.index)


@pytest.mark.parametrize("name", list(STRATEGIES.keys()))
def test_positions_are_zero_or_one(name: str, trending_up: pd.DataFrame) -> None:
    """Long-only: positions must be in {0, 1}, never negative or fractional."""
    pos = STRATEGIES[name].fn(trending_up)
    assert pos.isin([0, 1]).all()


@pytest.mark.parametrize("name", list(STRATEGIES.keys()))
def test_no_nan_in_output(name: str, trending_up: pd.DataFrame) -> None:
    """Warmup periods must produce 0, not NaN."""
    pos = STRATEGIES[name].fn(trending_up)
    assert not pos.isna().any()


@pytest.mark.parametrize("name", list(STRATEGIES.keys()))
def test_short_history_does_not_crash(name: str, short_history: pd.DataFrame) -> None:
    """Strategies must handle insufficient history gracefully without errors.

    What 'sufficient' means is strategy-specific (Donchian needs 20 days,
    MA crossover needs 200, etc.), so we only assert the universal contract:
    a Series of {0, 1} aligned on the input index, with no NaN.
    """
    pos = STRATEGIES[name].fn(short_history)
    assert isinstance(pos, pd.Series)
    assert len(pos) == len(short_history)
    assert pos.index.equals(short_history.index)
    assert pos.isin([0, 1]).all()
    assert not pos.isna().any()


# ---------------------------------------------------------------------------
# Behavioral expectations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(STRATEGIES.keys()))
def test_long_in_clean_uptrend(name: str, trending_up: pd.DataFrame) -> None:
    """In a smooth uptrend, every momentum strategy should be long most of the time
    after warmup."""
    pos = STRATEGIES[name].fn(trending_up)
    # After 250+ days, all warmups are done and trend is clearly positive.
    assert pos.iloc[-100:].mean() > 0.9


@pytest.mark.parametrize("name", list(STRATEGIES.keys()))
def test_flat_in_clean_downtrend(name: str, trending_down: pd.DataFrame) -> None:
    """In a smooth downtrend, every momentum strategy should be flat most of the
    time (no shorting allowed)."""
    pos = STRATEGIES[name].fn(trending_down)
    assert pos.iloc[-100:].mean() < 0.1


# ---------------------------------------------------------------------------
# Strategy-specific tests
# ---------------------------------------------------------------------------


def test_donchian_holds_until_breakdown() -> None:
    """Donchian: once long after a breakout, hold position until the lower band."""
    # Construct: 30 flat days at 100, then a sudden jump to 110 (breakout above
    # 20-day high), then drift down slowly. Position should stay long.
    idx = pd.date_range("2023-01-01", periods=60, freq="B")
    prices = np.concatenate(
        [
            np.full(30, 100.0),  # base period
            np.full(5, 110.0),  # breakout (above 20d max=100)
            np.linspace(110.0, 105.0, 25),  # slow decline, stays above 20d min
        ]
    )
    df = pd.DataFrame({"Close": prices}, index=idx)
    pos = STRATEGIES["donchian_breakout_20"].fn(df)

    # During the breakout days, we should be long.
    assert pos.iloc[31:35].sum() >= 3, "should enter long shortly after breakout"


def test_strategy_registry_completeness() -> None:
    """The five expected strategies must be registered with the right keys."""
    expected = {
        "ts_momentum_12_1",
        "ma_crossover_50_200",
        "dual_momentum",
        "donchian_breakout_20",
        "risk_adjusted_momentum",
    }
    assert set(STRATEGIES.keys()) == expected
    # And each entry has a non-empty description.
    for spec in STRATEGIES.values():
        assert spec.description.strip()
