from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from quant_agent.strategies.momentum import STRATEGIES
from quant_agent.tools.base import Tool


class ListStrategiesInput(BaseModel):
    """No parameters — this tool simply returns the registry."""


class ListStrategiesTool(Tool):
    name = "list_strategies"
    description = (
        "List all available momentum strategies with their names and descriptions. "
        "Call this first if the user asks to compare strategies and you don't yet "
        "know which ones are available."
    )
    input_model = ListStrategiesInput

    def run(self, params: ListStrategiesInput) -> dict[str, Any]:
        return {
            "strategies": [
                {"name": spec.name, "description": spec.description} for spec in STRATEGIES.values()
            ]
        }
