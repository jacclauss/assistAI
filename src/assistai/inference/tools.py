"""First-party tools and the surface the agent loop talks to.

The broker is the only production path to a handler, including the terminal
REPL. ``ToolRegistry`` is the catalog the broker wraps; tests talk to it
directly. ``get_time`` stays side-effect free.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import structlog

from assistai.clock import HouseholdClock, local_text, say_day, weekday_of
from assistai.errors import (
    CalendarError,
    JobError,
    MailError,
    RelayError,
    ResearchError,
    SharedError,
)
from assistai.inference.types import ToolCall, ToolSpec

log = structlog.get_logger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[str] | str]


@dataclass(frozen=True)
class ToolResult:
    """What the model sees, plus whether it came from outside the household."""

    content: str
    untrusted: bool = False


class ToolSurface(Protocol):
    """What ``run_turn`` needs. The broker is the production implementation."""

    def specs(self) -> list[ToolSpec]: ...

    async def execute(self, call: ToolCall) -> ToolResult: ...


GET_TIME_SPEC = ToolSpec(
    name="get_time",
    description=(
        "Return the current household local time, weekday, today's date, and UTC. "
        "Call this in the same turn before any at in job_create, job_reschedule, "
        "or relay. Use it to check which day 'tomorrow' or a weekday is; at still "
        "takes the person's words, not a time you work out."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)


class ToolRegistry:
    """Name → spec + handler. Unknown names return an error string, not an exception."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def names(self) -> frozenset[str]:
        return frozenset(self._specs)

    def spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    async def execute(self, call: ToolCall) -> ToolResult:
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolResult(json.dumps({"error": "unknown_tool", "name": call.name}))
        try:
            arguments = json.loads(call.arguments) if call.arguments else {}
        except json.JSONDecodeError:
            return ToolResult(json.dumps({"error": "invalid_arguments"}))
        if not isinstance(arguments, dict):
            return ToolResult(json.dumps({"error": "invalid_arguments"}))
        # A raising handler must not escape as a bare exception: the turn would
        # abort with an unanswered tool call already in history, which the
        # provider rejects on every later message.
        try:
            result = handler(arguments)
            content = result if isinstance(result, str) else await result
        except (JobError, RelayError, ResearchError, CalendarError, MailError, SharedError) as exc:
            log.exception("tool.failed", name=call.name)
            return ToolResult(
                json.dumps(
                    {"error": "tool_failed", "name": call.name, "message": str(exc)},
                    ensure_ascii=False,
                )
            )
        except Exception:
            log.exception("tool.failed", name=call.name)
            return ToolResult(json.dumps({"error": "tool_failed", "name": call.name}))
        return ToolResult(content)


def default_registry() -> ToolRegistry:
    """Unbrokered catalog for registry tests. Production paths use the broker."""
    registry = ToolRegistry()
    registry.register(GET_TIME_SPEC, get_time)
    return registry


def get_time(_arguments: dict[str, Any]) -> str:
    return json.dumps({"utc": datetime.now(UTC).isoformat()})


def bind_clock(
    timezone: str, *, now: Callable[[], float] | None = None
) -> Callable[[dict[str, Any]], str]:
    """Clock the model must read before it names an ``at``.

    The weekday and today's date are spelled out by the server, so the model
    does not work out which day "Monday" or "tomorrow" is from a timestamp.
    ``now`` is the wall clock unless an eval case stops it at its own moment.
    """

    def handler(_arguments: dict[str, Any]) -> str:
        # One reading for every field, so local, today, and utc cannot
        # straddle midnight and disagree about the day.
        instant = now() if now is not None else time.time()
        clock = HouseholdClock(zone=ZoneInfo(timezone), now=lambda: instant)
        today = clock.today()
        return json.dumps(
            {
                "timezone": timezone,
                "local": local_text(instant, clock=clock),
                "weekday": weekday_of(today),
                "today": say_day(today, clock=clock),
                "utc": datetime.fromtimestamp(instant, UTC).isoformat(),
            }
        )

    return handler
