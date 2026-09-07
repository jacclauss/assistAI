from __future__ import annotations

import httpx
import pytest

from assistai.errors import ToolLoopError
from assistai.inference.loop import run_turn
from assistai.inference.tools import GET_TIME_SPEC, ToolResult, default_registry
from assistai.inference.types import Message, ToolCall, ToolSpec
from tests.fakes import (
    client_for,
    completion_stream,
    pin,
    sequence,
    text_event,
    tool_event,
)


async def test_text_only_turn() -> None:
    client = client_for(lambda _req: completion_stream(text_event("hi", finish="stop")))
    messages = [Message(role="user", content="hello")]

    await run_turn(client, pin(), messages, None, max_tool_rounds=2)

    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[-1].content == "hi"
    await client.aclose()


async def test_tool_then_final_answer() -> None:
    client = client_for(
        sequence(
            completion_stream(
                tool_event(call_id="call_1", name="get_time", arguments="{}", finish="tool_calls")
            ),
            completion_stream(text_event("It is noon UTC.", finish="stop")),
        )
    )
    messages = [Message(role="user", content="what time is it?")]

    await run_turn(client, pin(), messages, default_registry(), max_tool_rounds=2)

    roles = [message.role for message in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert messages[1].tool_calls[0].name == "get_time"
    assert messages[2].tool_call_id == "call_1"
    assert "utc" in (messages[2].content or "")
    assert messages[-1].content == "It is noon UTC."
    assert messages[2].untrusted is False
    await client.aclose()


async def test_untrusted_results_are_marked_in_history() -> None:
    """The mark is what lets the next turn know the conversation is tainted."""

    class Untrusting:
        def specs(self) -> list[ToolSpec]:
            return [GET_TIME_SPEC]

        async def execute(self, call: ToolCall) -> ToolResult:
            return ToolResult('{"page": "from the open internet"}', untrusted=True)

    client = client_for(
        sequence(
            completion_stream(
                tool_event(call_id="call_1", name="get_time", arguments="{}", finish="tool_calls")
            ),
            completion_stream(text_event("done", finish="stop")),
        )
    )
    messages = [Message(role="user", content="look it up")]

    await run_turn(client, pin(), messages, Untrusting(), max_tool_rounds=2)

    tool_message = next(message for message in messages if message.role == "tool")
    assert tool_message.untrusted is True
    # The summary is derived from the page, so trimming the raw result must not
    # quietly clear the taint.
    assert messages[-1].role == "assistant"
    assert messages[-1].untrusted is True
    assert messages[1].untrusted is False
    await client.aclose()


async def test_a_later_turn_inherits_taint_from_context() -> None:
    """Trimming the fetched page must not launder the model's summary of it."""
    client = client_for(lambda _req: completion_stream(text_event("sure", finish="stop")))
    messages = [
        Message(role="user", content="look it up"),
        Message(role="tool", content="untrusted page", tool_call_id="c1", untrusted=True),
    ]

    await run_turn(client, pin(), messages, None, max_tool_rounds=2)

    assert messages[-1].role == "assistant"
    assert messages[-1].untrusted is True
    await client.aclose()


async def test_unknown_tool_returns_error_to_model() -> None:
    client = client_for(
        sequence(
            completion_stream(
                tool_event(call_id="call_9", name="explode", arguments="{}", finish="tool_calls")
            ),
            completion_stream(text_event("I cannot do that.", finish="stop")),
        )
    )
    messages = [Message(role="user", content="explode")]

    await run_turn(client, pin(), messages, default_registry(), max_tool_rounds=2)

    assert "unknown_tool" in (messages[2].content or "")
    await client.aclose()


async def test_max_rounds_fails_closed() -> None:
    def always_tool(_req: httpx.Request) -> httpx.Response:
        return completion_stream(
            tool_event(call_id="call_1", name="get_time", arguments="{}", finish="tool_calls")
        )

    client = client_for(always_tool)
    messages = [Message(role="user", content="loop")]

    with pytest.raises(ToolLoopError, match="max_tool_rounds"):
        await run_turn(client, pin(), messages, default_registry(), max_tool_rounds=1)

    await client.aclose()


async def test_tool_request_without_registry() -> None:
    client = client_for(
        lambda _req: completion_stream(
            tool_event(call_id="call_1", name="get_time", arguments="{}", finish="tool_calls")
        )
    )
    messages = [Message(role="user", content="time")]

    with pytest.raises(ToolLoopError, match="no registry"):
        await run_turn(client, pin(), messages, None, max_tool_rounds=2)

    await client.aclose()


async def test_history_is_preserved_across_turns() -> None:
    client = client_for(
        sequence(
            completion_stream(text_event("one", finish="stop")),
            completion_stream(text_event("two", finish="stop")),
        )
    )
    messages = [Message(role="user", content="first")]
    await run_turn(client, pin(), messages, None, max_tool_rounds=1)
    messages.append(Message(role="user", content="second"))
    await run_turn(client, pin(), messages, None, max_tool_rounds=1)

    assert [message.content for message in messages] == ["first", "one", "second", "two"]
    await client.aclose()
