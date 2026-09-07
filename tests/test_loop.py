from __future__ import annotations

import httpx
import pytest

from assistai.errors import ToolLoopError
from assistai.inference.loop import run_turn
from assistai.inference.tools import default_registry
from assistai.inference.types import Message
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
