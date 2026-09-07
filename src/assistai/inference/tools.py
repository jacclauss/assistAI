"""First-party tools available in the phase-1 loop.

The broker and per-agent ACLs arrive in phase 3. Until then the REPL may use
``get_time`` so the tool-call plumbing can be exercised against a real model.
The tool is side-effect free.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from assistai.inference.types import ToolCall, ToolSpec

ToolHandler = Callable[[dict[str, Any]], Awaitable[str] | str]


class ToolRegistry:
    """Name → spec + handler. Unknown names return an error string, not an exception."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    async def execute(self, call: ToolCall) -> str:
        handler = self._handlers.get(call.name)
        if handler is None:
            return json.dumps({"error": "unknown_tool", "name": call.name})
        try:
            arguments = json.loads(call.arguments) if call.arguments else {}
        except json.JSONDecodeError:
            return json.dumps({"error": "invalid_arguments"})
        if not isinstance(arguments, dict):
            return json.dumps({"error": "invalid_arguments"})
        result = handler(arguments)
        if isinstance(result, str):
            return result
        return await result


def default_registry() -> ToolRegistry:
    """The phase-1 REPL tool set: clock only."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_time",
            description=(
                "Return the current UTC time as an ISO-8601 timestamp. "
                "Use this when the user asks what time it is."
            ),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        _get_time,
    )
    return registry


def _get_time(_arguments: dict[str, Any]) -> str:
    return json.dumps({"utc": datetime.now(UTC).isoformat()})
