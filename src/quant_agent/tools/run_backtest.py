from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from quant_agent.backtest.engine import run_backtest
from quant_agent.data.cache import get_prices
from quant_agent.strategies.momentum import STRATEGIES
from quant_agent.tools.base import Tool


class RunBacktestInput(BaseModel):
    strategy: str = Field(description="Strategy name. Use list_strategies to see available names.")
    ticker: str = Field(default="SPY", description="Ticker symbol. Default 'SPY'.")
    start: str = Field(description="Start date YYYY-MM-DD.")
    end: str = Field(description="End date YYYY-MM-DD.")
    cost_bps: float = Field(
        default=1.0,
        description="Round-trip transaction cost in basis points per unit of turnover.",
    )


class RunBacktestTool(Tool):
    name = "run_backtest"
    description = (
        "Run a backtest of a named momentum strategy on a ticker over a date range. "
        "Returns a backtest_id (string) and a high-level summary. Pass the "
        "backtest_id to compute_stats and plot_results to drill in further."
    )
    input_model = RunBacktestInput

    def __init__(self, workspace: dict[str, Any]) -> None:
        """workspace is a shared dict where heavy result objects are stored."""
        self.workspace = workspace

    def run(self, params: RunBacktestInput) -> dict[str, Any]:
        if params.strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{params.strategy}'. Available: {list(STRATEGIES.keys())}"
            )

        prices = get_prices(params.ticker, params.start, params.end)
        positions = STRATEGIES[params.strategy].fn(prices)
        result = run_backtest(prices, positions, cost_bps=params.cost_bps)

        backtest_id = f"{params.strategy}__{params.ticker}__{params.start}__{params.end}"
        self.workspace[backtest_id] = result

        return {
            "backtest_id": backtest_id,
            "strategy": params.strategy,
            "ticker": params.ticker,
            "period": f"{params.start} to {params.end}",
            **result.summary(),
        }
