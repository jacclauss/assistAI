from __future__ import annotations

import pytest

from assistai.inference.tools import GET_TIME_SPEC, default_registry
from tests.fakes import body, tool_call


async def test_get_time_returns_utc() -> None:
    registry = default_registry()

    result = await body(registry, tool_call())

    assert "T" in result["utc"]
    assert result["utc"].endswith("+00:00") or result["utc"].endswith("Z")


async def test_unknown_tool_is_an_error_string() -> None:
    registry = default_registry()

    result = await body(registry, tool_call(name="shell"))

    assert result["error"] == "unknown_tool"
    assert result["name"] == "shell"


async def test_invalid_arguments_are_an_error_string() -> None:
    registry = default_registry()

    result = await body(registry, tool_call(arguments="{nope"))

    assert result["error"] == "invalid_arguments"


async def test_builtin_results_are_trusted() -> None:
    """Only outside content taints. A local clock must not poison the scope."""
    registry = default_registry()

    result = await registry.execute(tool_call())

    assert result.untrusted is False


@pytest.mark.parametrize("failing", ["sync", "async"])
async def test_a_raising_handler_becomes_a_tool_error(failing: str) -> None:
    """A handler that throws must not abort the turn.

    An escaping exception leaves the assistant's tool call unanswered in
    history, and every later request on that conversation is rejected for it.
    """
    registry = default_registry()

    def boom_sync(_arguments: dict[str, object]) -> str:
        raise RuntimeError("fw-secret leaked into the message")

    async def boom_async(_arguments: dict[str, object]) -> str:
        raise RuntimeError("fw-secret leaked into the message")

    registry.register(
        GET_TIME_SPEC,
        boom_sync if failing == "sync" else boom_async,
    )

    result = await body(registry, tool_call())

    assert result["error"] == "tool_failed"
    assert result["name"] == "get_time"


async def test_handler_failures_do_not_echo_the_exception() -> None:
    """The model is untrusted context. Exception text can carry secrets."""
    registry = default_registry()
    registry.register(
        GET_TIME_SPEC,
        lambda _a: (_ for _ in ()).throw(RuntimeError("token=fw-secret")),
    )

    result = await registry.execute(tool_call())

    assert "fw-secret" not in result.content
