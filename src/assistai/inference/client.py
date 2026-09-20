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
from assistai.inference.types import Completion, Message, TextDelta, ToolCall, ToolSpec, Usage
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
        try:
            response = await self._http.post(
                f"{self._settings.fireworks_base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise InferenceError(f"Fireworks is unreachable ({type(exc).__name__})") from exc
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
            "messages": [
                message.to_openai(nonce=self._settings.untrusted_nonce) for message in messages
            ],
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
            "stream": True,
            # Ask for a final usage frame so spend is observable per turn.
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = [tool.to_openai() for tool in tools]
        if pin.thinking:
            body["thinking"] = {"type": pin.thinking}

        url = f"{self._settings.fireworks_base_url.rstrip('/')}/chat/completions"
        try:
            # A streaming request, not a buffered post: on_delta must fire as
            # tokens arrive, and the read timeout should apply per chunk rather
            # than to the whole generation.
            async with self._http.stream(
                "POST", url, headers=self._headers(), json=body
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    if response.status_code in {401, 403}:
                        raise InferenceError("Fireworks rejected the API key")
                    if response.status_code == 404:
                        raise ModelNotAvailableError(f"Fireworks does not serve {pin.ref}")
                    raise InferenceError(_http_error(response))
                try:
                    return await _accumulate_stream(response, on_delta=on_delta)
                except ValueError as exc:
                    raise InferenceError(str(exc)) from exc
        except httpx.HTTPError as exc:
            # Timeouts and dropped connections are the common failure on a home
            # network. Speak AssistAIError so callers can fall back.
            raise InferenceError(f"Fireworks is unreachable ({type(exc).__name__})") from exc

    async def complete_json(self, pin: ModelPin, messages: list[Message]) -> dict[str, Any]:
        """Non-streaming JSON object completion for the quarantined summarizer."""
        body: dict[str, Any] = {
            "model": pin.ref,
            "messages": [
                message.to_openai(nonce=self._settings.untrusted_nonce) for message in messages
            ],
            "max_tokens": min(self._settings.max_tokens, 1024),
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        url = f"{self._settings.fireworks_base_url.rstrip('/')}/chat/completions"
        try:
            response = await self._http.post(url, headers=self._headers(), json=body)
        except httpx.HTTPError as exc:
            raise InferenceError(f"Fireworks is unreachable ({type(exc).__name__})") from exc
        if response.status_code in {401, 403}:
            raise InferenceError("Fireworks rejected the API key")
        if response.status_code == 404:
            raise ModelNotAvailableError(f"Fireworks does not serve {pin.ref}")
        if response.status_code >= 400:
            raise InferenceError(_http_error(response))
        try:
            payload: object = response.json()
        except json.JSONDecodeError as exc:
            raise InferenceError("Fireworks returned invalid JSON") from exc
        text = _json_message(payload)
        try:
            return _parse_json_object(text)
        except json.JSONDecodeError as exc:
            raise InferenceError("Fireworks JSON object was not valid JSON") from exc

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


def _json_message(payload: object) -> str:
    if not isinstance(payload, dict):
        raise InferenceError("Fireworks JSON object was not valid JSON")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise InferenceError("Fireworks JSON object was empty")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise InferenceError("Fireworks JSON object was empty")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise InferenceError("Fireworks JSON object was empty")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise InferenceError("Fireworks JSON object was empty")
    return content


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")]
        cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    payload: object = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("expected object", cleaned, 0)
    return payload


class _StreamState:
    """Mutable accumulator threaded through the SSE frames of one response."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.tools: dict[int, dict[str, str]] = {}
        self.finish_reason: str | None = None
        self.usage: Usage | None = None


async def _accumulate_stream(
    response: httpx.Response,
    *,
    on_delta: DeltaCallback | None,
) -> Completion:
    buffer = SSEBuffer()
    state = _StreamState()

    async for chunk in response.aiter_text():
        if _ingest(buffer, chunk, state, on_delta):
            return _finish(state)
    _ingest(buffer, None, state, on_delta)
    return _finish(state)


def _ingest(
    buffer: SSEBuffer,
    chunk: str | None,
    state: _StreamState,
    on_delta: DeltaCallback | None,
) -> bool:
    payloads = buffer.push(chunk) if chunk is not None else buffer.flush()
    for payload in payloads:
        if payload == DONE:
            return True
        event = parse_sse_json(payload)
        if event is None:
            continue
        _apply_event(event, state, on_delta)
    return False


def _apply_event(
    event: dict[str, Any],
    state: _StreamState,
    on_delta: DeltaCallback | None,
) -> None:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message", "provider error")
        raise InferenceError(f"Fireworks stream error: {message}")
    usage = _parse_usage(event.get("usage"))
    if usage is not None:
        state.usage = usage
    choices = event.get("choices")
    # The usage frame carries an empty choices list; that is not an error.
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, dict):
        return
    delta = choice.get("delta") or {}
    if isinstance(delta, dict):
        text = delta.get("content")
        # reasoning_content is ignored on purpose: thinking must not leak to the user.
        if isinstance(text, str) and text:
            state.content.append(text)
            if on_delta is not None:
                on_delta(TextDelta(text=text))
        _accumulate_tool_deltas(delta.get("tool_calls"), state.tools)
    reason = choice.get("finish_reason")
    if isinstance(reason, str):
        state.finish_reason = reason


def _parse_usage(raw: object) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens")
    completion = raw.get("completion_tokens")
    if not isinstance(prompt, int) and not isinstance(completion, int):
        return None
    return Usage(
        prompt_tokens=prompt if isinstance(prompt, int) else 0,
        completion_tokens=completion if isinstance(completion, int) else 0,
    )


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


def _finish(state: _StreamState) -> Completion:
    calls: list[ToolCall] = []
    for index in sorted(state.tools):
        slot = state.tools[index]
        if not slot["name"]:
            raise InferenceError(f"streamed tool call {index} is missing a name")
        if not slot["id"]:
            raise InferenceError(f"streamed tool call {index} is missing an id")
        _assert_json_object(slot["arguments"], index)
        calls.append(
            ToolCall(id=slot["id"], name=slot["name"], arguments=slot["arguments"] or "{}")
        )
    return Completion(
        content="".join(state.content),
        tool_calls=calls,
        finish_reason=state.finish_reason,
        usage=state.usage,
    )


def _assert_json_object(arguments: str, index: int) -> None:
    if not arguments:
        return
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise InferenceError(f"tool call {index} arguments are not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise InferenceError(f"tool call {index} arguments must be a JSON object")
