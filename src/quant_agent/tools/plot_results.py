"""Tool: plot_results — render equity curves to a PNG file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, required when no display is available
import matplotlib.pyplot as plt  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from quant_agent.tools.base import Tool  # noqa: E402


class PlotResultsInput(BaseModel):
    backtest_ids: list[str] = Field(
        description="One or more backtest_ids to plot together on the same chart."
    )
    output_path: str = Field(
        default="equity_curves.png",
        description="Where to save the PNG. Relative paths are fine.",
    )


class PlotResultsTool(Tool):
    name = "plot_results"
    description = (
        "Plot the equity curves of one or more backtests against the buy-and-hold "
        "benchmark. Saves a PNG file and returns its absolute path. Useful as the "
        "final step when comparing multiple strategies."
    )
    input_model = PlotResultsInput

    def __init__(self, workspace: dict[str, Any]) -> None:
        self.workspace = workspace

    def run(self, params: PlotResultsInput) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(10, 6))

        benchmark_drawn = False
        for bt_id in params.backtest_ids:
            if bt_id not in self.workspace:
                plt.close(fig)
                raise KeyError(f"Unknown backtest_id '{bt_id}'.")
            r = self.workspace[bt_id]
            ax.plot(r.equity.index, r.equity.values, label=bt_id, linewidth=1.2)
            if not benchmark_drawn:
                ax.plot(
                    r.benchmark_equity.index,
                    r.benchmark_equity.values,
                    label="Buy & Hold",
                    linestyle="--",
                    color="black",
                    linewidth=1,
                )
                benchmark_drawn = True

        ax.set_title("Strategy equity curves vs buy & hold")
        ax.set_ylabel("Equity (start = 1.0)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out = Path(params.output_path).resolve()
        fig.savefig(out, dpi=120)
        plt.close(fig)

        return {"path": str(out), "n_curves": len(params.backtest_ids)}
