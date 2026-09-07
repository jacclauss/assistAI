from __future__ import annotations

import httpx

from assistai.compare import compare_models, format_report
from assistai.manifest import ModelPin
from tests.fakes import (
    client_for,
    completion_stream,
    manifest,
    pin,
    sequence,
    settings,
    text_event,
    tool_event,
)


async def test_compare_scores_a_valid_tool_call() -> None:
    client = client_for(
        sequence(
            completion_stream(
                tool_event(call_id="c1", name="get_time", arguments="{}", finish="tool_calls")
            ),
            completion_stream(text_event("It is noon.", finish="stop")),
        )
    )

    results = await compare_models(
        settings(),
        manifest(),
        client=client,
        pins=(pin(),),
    )

    assert len(results) == 1
    assert results[0].available is True
    assert results[0].tool_called is True
    assert results[0].arguments_valid is True
    assert results[0].finished is True
    assert "yes" in format_report(results)
    await client.aclose()


async def test_compare_records_a_model_that_never_calls_the_tool() -> None:
    client = client_for(lambda _req: completion_stream(text_event("no idea", finish="stop")))

    results = await compare_models(
        settings(),
        manifest(),
        client=client,
        pins=(pin(),),
    )

    assert results[0].tool_called is False
    assert results[0].error == "no_tool_call"
    await client.aclose()


async def test_compare_continues_after_an_unreachable_candidate() -> None:
    def routing(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        if "broken-model" in body:
            raise httpx.ConnectError("down")
        return completion_stream(text_event("nope", finish="stop"))

    client = client_for(routing)
    pins = (
        ModelPin(name="broken", provider="fireworks", ref="accounts/fireworks/models/broken-model"),
        pin(),
    )

    results = await compare_models(settings(), manifest(), client=client, pins=pins)

    assert results[0].available is False
    assert results[0].error == "InferenceError"
    assert results[1].available is True
    assert results[1].tool_called is False
    await client.aclose()


async def test_an_unreachable_model_is_not_retried() -> None:
    """A ref the account cannot serve will not heal, and retries cost money."""
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectError("down")

    client = client_for(handler)

    results = await compare_models(settings(), manifest(), client=client, pins=(pin(),))

    assert attempts["n"] == 1
    assert results[0].available is False
    await client.aclose()


async def test_a_tool_loop_failure_is_not_an_unreachable_model() -> None:
    """These call for different fixes: change the manifest, or change the model.

    A model that loops forever on tool calls is reachable and answering. Filing
    it under 'unavailable' sends the operator to check the wrong thing.
    """
    client = client_for(
        lambda _req: completion_stream(
            tool_event(call_id="c1", name="get_time", arguments="{}", finish="tool_calls")
        )
    )

    results = await compare_models(
        settings(max_tool_rounds=1),
        manifest(),
        client=client,
        pins=(pin(),),
    )

    assert results[0].available is True
    assert results[0].tool_called is True
    assert results[0].finished is False
    assert results[0].error == "ToolLoopError"
    assert "yes" not in format_report(results).splitlines()[-1].split()[1]
    await client.aclose()
