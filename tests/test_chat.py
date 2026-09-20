from __future__ import annotations

import json

from assistai.agents import BrokerPolicy
from assistai.broker import builtin_catalog
from assistai.chat import _broker, chat_once, chat_repl
from tests.fakes import (
    catalog_then_completions,
    client_for,
    completion_stream,
    manifest,
    recorded,
    sequence,
    settings,
    text_event,
    tool_event,
)


async def test_once_returns_final_text() -> None:
    client = client_for(lambda _req: completion_stream(text_event("pong", finish="stop")))
    chunks: list[str] = []

    text = await chat_once(
        settings(),
        "ping",
        client=client,
        manifest=manifest(),
        use_tools=False,
        write=chunks.append,
    )

    assert text == "pong"
    assert "pong" in "".join(chunks)
    await client.aclose()


async def test_once_goes_through_the_broker() -> None:
    """The REPL is not an unbrokered path to whatever the catalog contains."""
    handler, seen = recorded(lambda _req: completion_stream(text_event("pong", finish="stop")))
    client = client_for(handler)

    await chat_once(
        settings(),
        "ping",
        client=client,
        manifest=manifest(),
        write=lambda _s: None,
    )

    body = json.loads(seen[0].content)
    names = [tool["function"]["name"] for tool in body["tools"]]
    assert names == ["get_time"]
    assert "relay" not in names
    assert "agent 'repl'" in body["messages"][0]["content"]
    await client.aclose()


async def test_once_denies_a_tool_the_model_invents() -> None:
    """The broker, not the catalog, is what the REPL executes."""
    handler, seen = recorded(
        sequence(
            completion_stream(
                tool_event(call_id="c1", name="shell", arguments="{}", finish="tool_calls")
            ),
            completion_stream(text_event("I cannot do that.", finish="stop")),
        )
    )
    client = client_for(handler)

    await chat_once(
        settings(),
        "run a shell",
        client=client,
        manifest=manifest(),
        write=lambda _s: None,
    )

    follow_up = json.loads(seen[1].content)
    tool_messages = [message for message in follow_up["messages"] if message["role"] == "tool"]
    assert tool_messages
    assert '"error": "tool_denied"' in tool_messages[0]["content"]
    assert '"reason": "unknown"' in tool_messages[0]["content"]
    await client.aclose()


async def test_once_denies_a_catalog_tool_off_the_acl() -> None:
    """web_fetch in the catalog must not become callable from make chat."""
    catalog = builtin_catalog()
    assert catalog.spec("web_fetch") is not None
    handler, seen = recorded(
        sequence(
            completion_stream(
                tool_event(call_id="c1", name="web_fetch", arguments="{}", finish="tool_calls")
            ),
            completion_stream(text_event("I cannot do that.", finish="stop")),
        )
    )
    client = client_for(handler)

    await chat_once(
        settings(),
        "look this up",
        client=client,
        manifest=manifest(),
        catalog=catalog,
        write=lambda _s: None,
    )

    advertised = json.loads(seen[0].content)
    names = [tool["function"]["name"] for tool in advertised["tools"]]
    assert names == ["get_time"]
    follow_up = json.loads(seen[1].content)
    tool_messages = [message for message in follow_up["messages"] if message["role"] == "tool"]
    assert tool_messages
    assert '"error": "tool_denied"' in tool_messages[0]["content"]
    assert '"reason": "acl"' in tool_messages[0]["content"]
    await client.aclose()


def test_repl_broker_uses_household_taint_sinks() -> None:
    assert _broker()._policy.tainted_sinks_denied == BrokerPolicy.defaults().tainted_sinks_denied


async def test_repl_quit_and_reset() -> None:
    client = client_for(
        catalog_then_completions(
            (text_event("first", finish="stop"),),
            (text_event("second", finish="stop"),),
        )
    )
    prompts = iter(["hello", "/reset", "again", "/quit"])
    lines: list[str] = []

    await chat_repl(
        settings(),
        client=client,
        manifest=manifest(),
        use_tools=False,
        read_line=lambda _prompt: next(prompts),
        write=lambda s: None,
        writeln=lines.append,
    )

    assert any("history cleared" in line for line in lines)
    await client.aclose()


async def test_repl_eof_exits_cleanly() -> None:
    client = client_for(catalog_then_completions())

    def boom(_prompt: str) -> str:
        raise EOFError

    await chat_repl(
        settings(),
        client=client,
        manifest=manifest(),
        use_tools=False,
        read_line=boom,
        write=lambda s: None,
        writeln=lambda s: None,
    )
    await client.aclose()
