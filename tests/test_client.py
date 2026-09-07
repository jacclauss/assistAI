from __future__ import annotations

import json

import httpx
import pytest
from tests.fakes import (
    PRIMARY,
    QUARANTINE,
    catalog_response,
    client_for,
    completion_stream,
    pin,
    probe_ok,
    recorded,
    sequence,
    text_event,
    tool_event,
)

from assistai.config import Settings
from assistai.errors import InferenceError, MissingAPIKeyError, ModelNotAvailableError
from assistai.inference.client import FireworksClient
from assistai.inference.types import Message


def test_missing_key_fails_closed() -> None:
    with pytest.raises(MissingAPIKeyError):
        FireworksClient(Settings())


async def test_validate_catalog_hit_skips_probe() -> None:
    handler, seen = recorded(sequence(catalog_response([PRIMARY, QUARANTINE])))
    client = client_for(handler)

    await client.validate((PRIMARY, QUARANTINE))

    assert len(seen) == 1
    assert "models" in str(seen[0].url)
    await client.aclose()


async def test_validate_catalog_miss_probes() -> None:
    handler = sequence(catalog_response(["accounts/fireworks/models/other"]), probe_ok())
    client = client_for(handler)

    await client.validate((PRIMARY,))

    await client.aclose()


async def test_validate_catalog_failure_falls_through_to_probe() -> None:
    handler = sequence(catalog_response([], status=500), probe_ok())
    client = client_for(handler)

    await client.validate((PRIMARY,))

    await client.aclose()


async def test_probe_404_is_not_available() -> None:
    client = client_for(lambda _req: httpx.Response(404, text="missing"))

    with pytest.raises(ModelNotAvailableError, match=PRIMARY):
        await client.probe(PRIMARY)

    await client.aclose()


async def test_probe_401_does_not_echo_key() -> None:
    client = client_for(lambda _req: httpx.Response(401, text="nope fw-secret"))

    with pytest.raises(InferenceError, match="API key") as exc:
        await client.probe(PRIMARY)

    assert "fw-secret" not in str(exc.value)
    await client.aclose()


async def test_complete_streams_text() -> None:
    deltas: list[str] = []
    client = client_for(
        lambda _req: completion_stream(text_event("Hel"), text_event("lo", finish="stop"))
    )

    result = await client.complete(
        pin(),
        [Message(role="user", content="hi")],
        on_delta=lambda delta: deltas.append(delta.text),
    )

    assert result.content == "Hello"
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    assert deltas == ["Hel", "lo"]
    await client.aclose()


async def test_complete_accumulates_split_tool_arguments() -> None:
    client = client_for(
        lambda _req: completion_stream(
            tool_event(call_id="call_1", name="get_time", arguments=""),
            tool_event(arguments="{"),
            tool_event(arguments="}", finish="tool_calls"),
        )
    )

    result = await client.complete(pin(), [Message(role="user", content="time?")])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "get_time"
    assert result.tool_calls[0].arguments == "{}"
    await client.aclose()


async def test_complete_rejects_malformed_tool_json() -> None:
    client = client_for(
        lambda _req: completion_stream(
            tool_event(call_id="call_1", name="get_time", arguments="{nope", finish="tool_calls")
        )
    )

    with pytest.raises(InferenceError, match="valid JSON"):
        await client.complete(pin(), [Message(role="user", content="time?")])

    await client.aclose()


async def test_reasoning_content_is_not_surfaced() -> None:
    client = client_for(
        lambda _req: completion_stream(
            {"choices": [{"delta": {"reasoning_content": "secret chain", "content": "ok"}}]}
        )
    )

    result = await client.complete(pin(), [Message(role="user", content="hi")])

    assert result.content == "ok"
    assert "secret" not in result.content
    await client.aclose()


async def test_request_includes_tools_thinking_and_bearer() -> None:
    handler, seen = recorded(lambda _req: completion_stream(text_event("ok", finish="stop")))
    client = client_for(handler)

    await client.complete(
        pin(thinking="disabled"),
        [Message(role="user", content="hi")],
        tools=[],
    )
    # Re-send with a real tool spec so the body includes tools.
    from assistai.inference.tools import default_registry

    await client.complete(
        pin(thinking="disabled"),
        [Message(role="user", content="hi")],
        tools=default_registry().specs(),
    )

    request = seen[-1]
    assert request.headers["Authorization"] == "Bearer fw-secret"
    body = json.loads(request.content)
    assert body["model"] == PRIMARY
    assert body["stream"] is True
    assert body["thinking"] == {"type": "disabled"}
    assert body["tools"][0]["function"]["name"] == "get_time"
    await client.aclose()


async def test_http_error_body_truncated_and_keyless() -> None:
    client = client_for(lambda _req: httpx.Response(500, text="boom " + ("fw-secret " * 80)))

    with pytest.raises(InferenceError) as exc:
        await client.complete(pin(), [Message(role="user", content="hi")])

    message = str(exc.value)
    assert "500" in message
    assert len(message) < 400
    await client.aclose()


async def test_stream_error_object() -> None:
    client = client_for(lambda _req: completion_stream({"error": {"message": "overloaded"}}))

    with pytest.raises(InferenceError, match="overloaded"):
        await client.complete(pin(), [Message(role="user", content="hi")])

    await client.aclose()
