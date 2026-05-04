"""Tests for the 5 agent tools.

We test them in isolation (without invoking the LLM) to verify their
contract: input validation, output shape, and integration with the
shared workspace.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quant_agent.data import cache as cache_mod
from quant_agent.tools.compute_stats import ComputeStatsInput, ComputeStatsTool
from quant_agent.tools.fetch_prices import FetchPricesInput, FetchPricesTool
from quant_agent.tools.list_strategies import ListStrategiesInput, ListStrategiesTool
from quant_agent.tools.plot_results import PlotResultsInput, PlotResultsTool
from quant_agent.tools.registry import build_registry
from quant_agent.tools.run_backtest import RunBacktestInput, RunBacktestTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_prices() -> pd.DataFrame:
    """Realistic SPY-like price series, 300 business days."""
    idx = pd.date_range("2022-01-03", periods=300, freq="B")
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, size=300)
    close = 400 * (1 + returns).cumprod()
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.integers(50_000_000, 100_000_000, size=300),
        },
        index=idx,
    )


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cache_mod.settings, "cache_dir", tmp_path)
    return tmp_path


@pytest.fixture
def patched_download(fake_prices: pd.DataFrame, isolated_cache: Path):
    """yfinance is mocked to return the same fake prices every time."""
    with patch.object(cache_mod, "_download", return_value=fake_prices) as m:
        yield m


# ---------------------------------------------------------------------------
# list_strategies
# ---------------------------------------------------------------------------


def test_list_strategies_returns_all_five() -> None:
    out = ListStrategiesTool().run(ListStrategiesInput())
    assert len(out["strategies"]) == 5
    assert all("name" in s and "description" in s for s in out["strategies"])


# ---------------------------------------------------------------------------
# fetch_prices
# ---------------------------------------------------------------------------


def test_fetch_prices_returns_summary(patched_download) -> None:
    out = FetchPricesTool().run(FetchPricesInput(start="2022-01-01", end="2023-04-01"))
    assert out["ticker"] == "SPY"
    assert out["n_rows"] == 300
    assert isinstance(out["first_close"], float)


def test_fetch_prices_validates_dates() -> None:
    """Pydantic catches missing required fields."""
    with pytest.raises(ValidationError):
        FetchPricesInput(start="2022-01-01")  # missing 'end'


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------


def test_run_backtest_stores_in_workspace(patched_download) -> None:
    workspace: dict = {}
    tool = RunBacktestTool(workspace)
    out = tool.run(
        RunBacktestInput(
            strategy="ma_crossover_50_200",
            start="2022-01-01",
            end="2023-04-01",
        )
    )
    assert out["backtest_id"] in workspace
    assert out["n_days"] == 300
    assert "final_equity" in out


def test_run_backtest_unknown_strategy(patched_download) -> None:
    tool = RunBacktestTool({})
    with pytest.raises(ValueError, match="Unknown strategy"):
        tool.run(
            RunBacktestInput(
                strategy="not_a_strategy",
                start="2022-01-01",
                end="2023-04-01",
            )
        )


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


def test_compute_stats_shape(patched_download) -> None:
    workspace: dict = {}
    bt_tool = RunBacktestTool(workspace)
    out = bt_tool.run(
        RunBacktestInput(
            strategy="dual_momentum",
            start="2022-01-01",
            end="2023-04-01",
        )
    )
    bt_id = out["backtest_id"]

    stats = ComputeStatsTool(workspace).run(ComputeStatsInput(backtest_id=bt_id))
    expected_keys = {"sharpe", "sortino", "cagr", "vol_annualized", "max_drawdown", "hit_rate"}
    assert set(stats["strategy_stats"].keys()) == expected_keys
    assert set(stats["benchmark_stats"].keys()) == expected_keys


def test_compute_stats_unknown_backtest_id() -> None:
    with pytest.raises(KeyError, match="Unknown backtest_id"):
        ComputeStatsTool({}).run(ComputeStatsInput(backtest_id="not_real"))


# ---------------------------------------------------------------------------
# plot_results
# ---------------------------------------------------------------------------


def test_plot_results_writes_png(patched_download, tmp_path: Path) -> None:
    workspace: dict = {}
    bt_tool = RunBacktestTool(workspace)
    out = bt_tool.run(
        RunBacktestInput(
            strategy="ts_momentum_12_1",
            start="2022-01-01",
            end="2023-04-01",
        )
    )

    output_file = tmp_path / "equity.png"
    plot_out = PlotResultsTool(workspace).run(
        PlotResultsInput(
            backtest_ids=[out["backtest_id"]],
            output_path=str(output_file),
        )
    )

    assert Path(plot_out["path"]).exists()
    assert plot_out["n_curves"] == 1


# ---------------------------------------------------------------------------
# Registry — integration
# ---------------------------------------------------------------------------


def test_registry_has_all_five_tools() -> None:
    registry = build_registry()
    assert set(registry.names()) == {
        "list_strategies",
        "fetch_prices",
        "run_backtest",
        "compute_stats",
        "plot_results",
    }


def test_registry_anthropic_schemas_are_well_formed() -> None:
    """Each schema must have the keys the Anthropic API expects."""
    registry = build_registry()
    schemas = registry.anthropic_schemas()
    assert len(schemas) == 5
    for schema in schemas:
        assert set(schema.keys()) == {"name", "description", "input_schema"}
        assert isinstance(schema["name"], str)
        assert isinstance(schema["description"], str)
        assert "type" in schema["input_schema"]


def test_registry_execute_validates_input(patched_download) -> None:
    """Bad input must raise ValidationError, not crash deeper in the tool."""
    registry = build_registry()
    with pytest.raises(ValidationError):
        registry.execute("fetch_prices", {"ticker": "SPY"})  # missing start/end


def test_registry_full_pipeline(patched_download, tmp_path: Path) -> None:
    """End-to-end smoke test: list -> backtest -> stats -> plot."""
    workspace: dict = {}
    registry = build_registry(workspace)

    listed = registry.execute("list_strategies", {})
    assert len(listed["strategies"]) == 5

    bt = registry.execute(
        "run_backtest",
        {
            "strategy": "ma_crossover_50_200",
            "start": "2022-01-01",
            "end": "2023-04-01",
        },
    )
    bt_id = bt["backtest_id"]

    stats = registry.execute("compute_stats", {"backtest_id": bt_id})
    assert "strategy_stats" in stats

    plot = registry.execute(
        "plot_results",
        {
            "backtest_ids": [bt_id],
            "output_path": str(tmp_path / "out.png"),
        },
    )
    assert Path(plot["path"]).exists()
