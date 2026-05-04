"""Factory that assembles all tools into a ToolRegistry.

Centralizes the wiring between tools and the shared workspace.
"""

from __future__ import annotations

from typing import Any

from quant_agent.tools.base import ToolRegistry
from quant_agent.tools.compute_stats import ComputeStatsTool
from quant_agent.tools.fetch_prices import FetchPricesTool
from quant_agent.tools.list_strategies import ListStrategiesTool
from quant_agent.tools.plot_results import PlotResultsTool
from quant_agent.tools.run_backtest import RunBacktestTool


def build_registry(workspace: dict[str, Any] | None = None) -> ToolRegistry:
    """Build a registry with all 5 tools, sharing one workspace dict.

    Pass an existing workspace to inject state (useful for tests).
    Defaults to a fresh empty dict.
    """
    if workspace is None:
        workspace = {}

    registry = ToolRegistry()
    registry.register(ListStrategiesTool())
    registry.register(FetchPricesTool())
    registry.register(RunBacktestTool(workspace))
    registry.register(ComputeStatsTool(workspace))
    registry.register(PlotResultsTool(workspace))
    return registry
