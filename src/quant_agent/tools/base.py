"""Base abstractions for agent tools.

A tool is a typed wrapper around a Python function that the LLM can call.
Each tool defines:
- a name (snake_case, what the LLM uses to identify it)
- a description (natural language, what the LLM reads to decide when to use it)
- an input model (Pydantic, defines + validates the parameters)
- a run() method that executes the tool given validated parameters
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class Tool(ABC):
    """Abstract base class for all tools.

    Subclasses MUST set the three ClassVars below and implement run().
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, params: BaseModel) -> dict[str, Any]:
        """Execute the tool. Must return a JSON-serializable dict."""
        raise NotImplementedError

    @classmethod
    def to_anthropic_schema(cls) -> dict[str, Any]:
        """Convert to the schema format expected by the Anthropic API."""
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls.input_model.model_json_schema(),
        }


class ToolRegistry:
    """Holds the set of tools available to the agent.

    Acts as a single point of validation + dispatch: the LLM gives us a
    tool name and a raw input dict, the registry validates the dict against
    the tool's Pydantic model, then runs the tool. Validation errors are
    raised as ValueError so the agent can feed them back to the LLM.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def anthropic_schemas(self) -> list[dict[str, Any]]:
        """Schemas to pass to the Anthropic API as the `tools` parameter."""
        return [type(t).to_anthropic_schema() for t in self._tools.values()]

    def execute(self, name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
        """Validate input, dispatch, and return the result.

        Raises:
            KeyError: tool name not registered.
            ValidationError: input does not match the tool's schema.
        """
        tool = self.get(name)
        validated = tool.input_model.model_validate(raw_input)
        return tool.run(validated)
