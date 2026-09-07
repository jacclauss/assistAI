"""Shared Fireworks HTTP doubles for unit tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from assistai.config import Settings
from assistai.inference.client import FireworksClient
from assistai.inference.tools import ToolSurface
from assistai.inference.types import ToolCall
from assistai.manifest import Manifest, ModelPin

PRIMARY = "accounts/fireworks/models/minimax-m3"
QUARANTINE = "accounts/fireworks/models/deepseek-v4-flash-0731"


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "FIREWORKS_API_KEY": "fw-secret",
        "fireworks_base_url": "https://fw.test/inference/v1",
        "fireworks_account_url": "https://fw.test/v1",
    }
    values.update(overrides)
    return Settings(**values)


def pin(ref: str = PRIMARY, *, thinking: str | None = "disabled") -> ModelPin:
    return ModelPin(name="primary", provider="fireworks", ref=ref, thinking=thinking)


def manifest() -> Manifest:
    return Manifest(
        primary=pin(),
        quarantine=ModelPin(name="quarantine", provider="fireworks", ref=QUARANTINE),
        candidates={},
    )


def sse_frames(*events: dict[str, Any] | str) -> bytes:
    lines: list[str] = []
    for event in events:
        if isinstance(event, str):
            lines.append(f"data: {event}\n\n")
        else:
            lines.append(f"data: {json.dumps(event)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def text_event(text: str, *, finish: str | None = None) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "delta": {"content": text}}
    if finish is not None:
        choice["finish_reason"] = finish
    return {"choices": [choice]}


def tool_event(
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    finish: str | None = None,
) -> dict[str, Any]:
    function: dict[str, Any] = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    call: dict[str, Any] = {"index": index, "function": function}
    if call_id is not None:
        call["id"] = call_id
    choice: dict[str, Any] = {"index": 0, "delta": {"tool_calls": [call]}}
    if finish is not None:
        choice["finish_reason"] = finish
    return {"choices": [choice]}


Handler = Callable[[httpx.Request], httpx.Response]


def client_for(handler: Handler, **overrides: Any) -> FireworksClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return FireworksClient(settings(**overrides), http=http)


def sequence(*responses: httpx.Response | Handler) -> Handler:
    """Return each response in order. Extra calls raise."""
    leftover = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not leftover:
            raise AssertionError(f"unexpected extra request: {request.method} {request.url}")
        item = leftover.pop(0)
        if callable(item) and not isinstance(item, httpx.Response):
            return item(request)
        return item

    return handler


def catalog_response(names: Iterable[str], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={"models": [{"name": name} for name in names]},
    )


def probe_ok() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})


def completion_stream(*events: dict[str, Any] | str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=sse_frames(*events),
    )


def catalog_then_completions(*events_per_call: tuple[dict[str, Any] | str, ...]) -> Handler:
    """First GET /models is a full catalog hit; later POSTs stream the given events."""
    leftover = [completion_stream(*events) for events in events_per_call]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return catalog_response([PRIMARY, QUARANTINE])
        if leftover:
            return leftover.pop(0)
        raise AssertionError(f"unexpected extra request: {request.method} {request.url}")

    return handler


def recorded(handler: Handler) -> tuple[Handler, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return wrapped, seen


def tool_call(name: str = "get_time", arguments: str = "{}") -> ToolCall:
    return ToolCall(id="call_1", name=name, arguments=arguments)


async def body(surface: ToolSurface, call: ToolCall) -> dict[str, Any]:
    """Execute a call and decode what the model would see."""
    result = await surface.execute(call)
    parsed = json.loads(result.content)
    assert isinstance(parsed, dict)
    return parsed
