"""Tool: compute_stats — compute performance metrics for a backtest."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant_agent.tools.base import Tool

_ANN_FACTOR = 252


def _stats(returns: pd.Series) -> dict[str, float]:
    """Compute Sharpe, Sortino, CAGR, vol, max drawdown, hit rate."""
    if len(returns) < 2 or returns.std() == 0:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "cagr": 0.0,
            "vol_annualized": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
        }

    mean_ann = returns.mean() * _ANN_FACTOR
    vol_ann = returns.std() * np.sqrt(_ANN_FACTOR)
    sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0

    downside_vol = returns[returns < 0].std() * np.sqrt(_ANN_FACTOR)
    sortino = mean_ann / downside_vol if downside_vol > 0 else 0.0

    equity = (1 + returns).cumprod()
    n_years = len(returns) / _ANN_FACTOR
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    nonzero = returns[returns != 0]
    hit_rate = (nonzero > 0).sum() / len(nonzero) if len(nonzero) > 0 else 0.0

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "cagr": round(float(cagr), 4),
        "vol_annualized": round(float(vol_ann), 4),
        "max_drawdown": round(float(max_dd), 4),
        "hit_rate": round(float(hit_rate), 4),
    }


class ComputeStatsInput(BaseModel):
    backtest_id: str = Field(description="The backtest_id returned by run_backtest.")


class ComputeStatsTool(Tool):
    name = "compute_stats"
    description = (
        "Compute performance statistics for a backtest: Sharpe, Sortino, CAGR, "
        "annualized volatility, max drawdown, hit rate. Returns both the strategy "
        "stats and the buy-and-hold benchmark stats so they can be compared."
    )
    input_model = ComputeStatsInput

    def __init__(self, workspace: dict[str, Any]) -> None:
        self.workspace = workspace

    def run(self, params: ComputeStatsInput) -> dict[str, Any]:
        if params.backtest_id not in self.workspace:
            raise KeyError(f"Unknown backtest_id '{params.backtest_id}'. Run run_backtest first.")
        result = self.workspace[params.backtest_id]
        return {
            "backtest_id": params.backtest_id,
            "strategy_stats": _stats(result.returns),
            "benchmark_stats": _stats(result.benchmark_returns),
        }
