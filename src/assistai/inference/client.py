"""OpenAI-compatible Fireworks client.

The inference ``/v1/models`` catalog is known to omit callable models (MiniMax
M3 is a documented miss). Startup validation therefore treats the catalog as a
hint and confirms each pin with a one-token probe.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

import httpx
import structlog

from assistai.config import Settings
from assistai.errors import InferenceError, MissingAPIKeyError, ModelNotAvailableError
from assistai.inference.sse import DONE, SSEBuffer, parse_sse_json
from assistai.inference.types import Completion, Message, TextDelta, ToolCall, ToolSpec
from assistai.manifest import ModelPin

log = structlog.get_logger(__name__)

DeltaCallback = Callable[[TextDelta], None]


class FireworksClient:
    """Talk to Fireworks over HTTP. The API key never appears in logs or errors."""

    def __init__(
        self,
        settings: Settings,
        *,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        if settings.fireworks_api_key is None:
            raise MissingAPIKeyError("FIREWORKS_API_KEY is not set; copy .env.example to .env")
        self._settings = settings
        self._api_key = settings.fireworks_api_key.get_secret_value()
        self._owns_http = http is None
        timeout = httpx.Timeout(settings.request_timeout_seconds)
        self._http = http or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def validate(self, refs: Iterable[str]) -> None:
        """Fail closed if a required model ref is not callable.

        Catalog omissions are not failures. A failed probe is.
        """
        catalog = await self._try_catalog()
        for ref in refs:
            if catalog is not None and ref in catalog:
                log.info("model.catalog_hit", ref=ref)
                continue
            if catalog is not None:
                log.warning("model.catalog_miss_probing", ref=ref)
            await self.probe(ref)
            log.info("model.probe_ok", ref=ref)

    async def probe(self, ref: str) -> None:
        """One-token non-streaming completion. Confirms the ref is callable."""
        payload = {
            "model": ref,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
        response = await self._http.post(
            f"{self._settings.fireworks_base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        if response.status_code == 404:
            raise ModelNotAvailableError(f"Fireworks does not serve {ref}")
        if response.status_code in {401, 403}:
            raise InferenceError("Fireworks rejected the API key")
        if response.status_code >= 400:
            raise InferenceError(_http_error(response))

    async def complete(
        self,
        pin: ModelPin,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        on_delta: DeltaCallback | None = None,
    ) -> Completion:
        """Stream a chat completion and accumulate content plus tool calls."""
        body: dict[str, Any] = {
            "model": pin.ref,
            "messages": [message.to_openai() for message in messages],
            "max_tokens": self._settings.max_tokens,
            "temperature": 0.2,
            "stream": True,
        }
        if tools:
            body["tools"] = [tool.to_openai() for tool in tools]
        if pin.thinking:
            body["thinking"] = {"type": pin.thinking}

        response = await self._http.post(
            f"{self._settings.fireworks_base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=body,
        )
        if response.status_code >= 400:
            if response.status_code in {401, 403}:
                raise InferenceError("Fireworks rejected the API key")
            if response.status_code == 404:
                raise ModelNotAvailableError(f"Fireworks does not serve {pin.ref}")
            raise InferenceError(_http_error(response))

        try:
            return await _accumulate_stream(response, on_delta=on_delta)
        except ValueError as exc:
            raise InferenceError(str(exc)) from exc

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _try_catalog(self) -> set[str] | None:
        url = f"{self._settings.fireworks_account_url.rstrip('/')}/accounts/fireworks/models"
        try:
            response = await self._http.get(
                url,
                headers=self._headers(),
                params={"filter": "supports_serverless=true", "pageSize": "200"},
            )
        except httpx.HTTPError as exc:
            log.warning("model.catalog_unreachable", error=type(exc).__name__)
            return None
        if response.status_code >= 400:
            log.warning("model.catalog_http", status=response.status_code)
            return None
        try:
            payload = response.json()
        except json.JSONDecodeError:
            log.warning("model.catalog_not_json")
            return None
        return _catalog_names(payload)


def _catalog_names(payload: object) -> set[str]:
    names: set[str] = set()
    if not isinstance(payload, dict):
        return names
    rows = payload.get("models") or payload.get("data") or []
    if not isinstance(rows, list):
        return names
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("name", "id"):
            value = row.get(key)
            if isinstance(value, str) and value:
                names.add(value)
    return names


def _http_error(response: httpx.Response) -> str:
    body = response.text[:300].replace("\n", " ")
    return f"Fireworks HTTP {response.status_code}: {body}"


async def _accumulate_stream(
    response: httpx.Response,
    *,
    on_delta: DeltaCallback | None,
) -> Completion:
    buffer = SSEBuffer()
    content_parts: list[str] = []
    tool_acc: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None

    async for chunk in response.aiter_text():
        stop, finish_reason = _ingest(
            buffer, chunk, content_parts, tool_acc, on_delta, finish_reason
        )
        if stop:
            return _finish(content_parts, tool_acc, finish_reason)
    _ingest(buffer, None, content_parts, tool_acc, on_delta, finish_reason)
    return _finish(content_parts, tool_acc, finish_reason)


def _ingest(
    buffer: SSEBuffer,
    chunk: str | None,
    content_parts: list[str],
    tool_acc: dict[int, dict[str, str]],
    on_delta: DeltaCallback | None,
    finish_reason: str | None,
) -> tuple[bool, str | None]:
    payloads = buffer.push(chunk) if chunk is not None else buffer.flush()
    for payload in payloads:
        if payload == DONE:
            return True, finish_reason
        event = parse_sse_json(payload)
        if event is None:
            continue
        finish_reason = _apply_event(event, content_parts, tool_acc, on_delta) or finish_reason
    return False, finish_reason


def _apply_event(
    event: dict[str, Any],
    content_parts: list[str],
    tool_acc: dict[int, dict[str, str]],
    on_delta: DeltaCallback | None,
) -> str | None:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message", "provider error")
        raise InferenceError(f"Fireworks stream error: {message}")
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    delta = choice.get("delta") or {}
    if isinstance(delta, dict):
        text = delta.get("content")
        # reasoning_content is ignored on purpose: thinking must not leak to the user.
        if isinstance(text, str) and text:
            content_parts.append(text)
            if on_delta is not None:
                on_delta(TextDelta(text=text))
        _accumulate_tool_deltas(delta.get("tool_calls"), tool_acc)
    reason = choice.get("finish_reason")
    return reason if isinstance(reason, str) else None


def _accumulate_tool_deltas(deltas: object, tool_acc: dict[int, dict[str, str]]) -> None:
    if not isinstance(deltas, list):
        return
    for item in deltas:
        if not isinstance(item, dict):
            continue
        index = item.get("index", 0)
        if not isinstance(index, int):
            continue
        slot = tool_acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
        call_id = item.get("id")
        if isinstance(call_id, str) and call_id:
            slot["id"] = call_id
        function = item.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            slot["name"] = name
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            slot["arguments"] += arguments


def _finish(
    content_parts: list[str],
    tool_acc: dict[int, dict[str, str]],
    finish_reason: str | None,
) -> Completion:
    calls: list[ToolCall] = []
    for index in sorted(tool_acc):
        slot = tool_acc[index]
        if not slot["name"]:
            raise InferenceError(f"streamed tool call {index} is missing a name")
        if not slot["id"]:
            raise InferenceError(f"streamed tool call {index} is missing an id")
        _assert_json_object(slot["arguments"], index)
        calls.append(
            ToolCall(id=slot["id"], name=slot["name"], arguments=slot["arguments"] or "{}")
        )
    return Completion(content="".join(content_parts), tool_calls=calls, finish_reason=finish_reason)


def _assert_json_object(arguments: str, index: int) -> None:
    if not arguments:
        return
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise InferenceError(f"tool call {index} arguments are not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise InferenceError(f"tool call {index} arguments must be a JSON object")
