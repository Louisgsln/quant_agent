from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_agent.data import cache as cache_mod


@pytest.fixture
def fake_ohlcv() -> pd.DataFrame:
    """A small but realistic-looking OHLCV DataFrame."""
    idx = pd.date_range("2023-01-02", periods=10, freq="B")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, size=10))
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, size=10),
        },
        index=idx,
    )


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the cache to a temporary directory unique to each test."""
    monkeypatch.setattr(cache_mod.settings, "cache_dir", tmp_path)
    return tmp_path


def test_cache_miss_then_hit(fake_ohlcv: pd.DataFrame, isolated_cache: Path) -> None:
    """First call downloads and writes; second call reads from disk."""
    with patch.object(cache_mod, "_download", return_value=fake_ohlcv) as mock_dl:
        df1 = cache_mod.get_prices("SPY", "2023-01-01", "2023-01-15")
        df2 = cache_mod.get_prices("SPY", "2023-01-01", "2023-01-15")

    assert mock_dl.call_count == 1, "second call must hit cache, not re-download"
    pd.testing.assert_frame_equal(df1, df2)


def test_cache_key_separates_distinct_ranges(
    fake_ohlcv: pd.DataFrame, isolated_cache: Path
) -> None:
    """Different date ranges must not share a cache file."""
    with patch.object(cache_mod, "_download", return_value=fake_ohlcv) as mock_dl:
        cache_mod.get_prices("SPY", "2023-01-01", "2023-01-15")
        cache_mod.get_prices("SPY", "2023-01-01", "2023-02-15")

    assert mock_dl.call_count == 2


def test_use_cache_false_forces_redownload(
    fake_ohlcv: pd.DataFrame, isolated_cache: Path
) -> None:
    """Even with a cached file, use_cache=False bypasses it."""
    with patch.object(cache_mod, "_download", return_value=fake_ohlcv) as mock_dl:
        cache_mod.get_prices("SPY", "2023-01-01", "2023-01-15")
        cache_mod.get_prices("SPY", "2023-01-01", "2023-01-15", use_cache=False)

    assert mock_dl.call_count == 2


def test_index_is_tz_naive(fake_ohlcv: pd.DataFrame, isolated_cache: Path) -> None:
    """Cached data must have a timezone-naive DatetimeIndex."""
    with patch.object(cache_mod, "_download", return_value=fake_ohlcv):
        df = cache_mod.get_prices("SPY", "2023-01-01", "2023-01-15")

    assert df.index.tz is None