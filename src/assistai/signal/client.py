"""HTTP + WebSocket client for signal-cli-rest-api.

Send and health stay on HTTP. Receive is a WebSocket: json-rpc mode drops
messages that arrive while no client is connected, so polling is not a fallback.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, suppress
from typing import Any, Protocol
from urllib.parse import quote, urlparse, urlunparse

import httpx
import structlog
from websockets.asyncio.client import connect as ws_connect

from assistai.config import Settings
from assistai.errors import SignalError, SignalUnavailableError

log = structlog.get_logger(__name__)

# Older Signal clients truncate around 2 KiB. Stay under that and send parts.
_MAX_SEND_CHARS = 1900


class WebSocketConnection(Protocol):
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    async def close(self) -> None: ...


ConnectFn = Callable[[str], AbstractAsyncContextManager[WebSocketConnection]]


class SignalTransport(Protocol):
    """The subset of SignalClient the channel depends on. Tests supply fakes."""

    async def send(self, recipient: str, text: str) -> None: ...

    def receive(self, stop: asyncio.Event) -> AsyncIterator[object]: ...

    async def check(self) -> None: ...

    async def wait_until_healthy(
        self, timeout_seconds: float, stop: asyncio.Event, *, poll_seconds: float = 3.0
    ) -> None: ...

    async def aclose(self) -> None: ...


class SignalClient:
    """Talk to signal-cli-rest-api. No SDK; we own the wire format."""

    def __init__(
        self,
        settings: Settings,
        *,
        http: httpx.AsyncClient | None = None,
        connect: ConnectFn | None = None,
    ) -> None:
        self._settings = settings
        self._base = settings.signal_base_url.rstrip("/")
        self._account = settings.signal_account
        self._owns_http = http is None
        timeout = httpx.Timeout(settings.request_timeout_seconds)
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._connect: ConnectFn = connect or _default_connect

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def wait_until_healthy(
        self,
        timeout_seconds: float,
        stop: asyncio.Event,
        *,
        poll_seconds: float = 3.0,
    ) -> None:
        """Poll /v1/health until it answers.

        signal-cli in json-rpc mode boots slowly, so a cold ``docker compose
        up`` or a Pi reboot would otherwise exit before it is listening and let
        the container runtime restart-loop the gateway.
        """
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        attempt = 0
        while True:
            try:
                await self._health()
                return
            except SignalUnavailableError:
                attempt += 1
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0 or stop.is_set():
                    raise
                log.info("signal.waiting_for_cli", attempt=attempt)
                with suppress(TimeoutError):
                    await _wait(stop, min(poll_seconds, remaining))
                if stop.is_set():
                    raise

    async def check(self) -> None:
        """Fail closed if signal-cli is down or the account is not registered."""
        await self._health()
        about = await self._get(f"{self._base}/v1/about")
        mode = _about_mode(about)
        if mode is not None and mode != "json-rpc":
            log.warning("signal.wrong_mode", mode=mode)
        if self._account:
            accounts = await self._list_accounts()
            if accounts is not None and self._account not in accounts:
                # Receiving would reconnect-loop forever against an account that
                # does not exist. Say so now, while the operator is watching.
                raise SignalUnavailableError(
                    f"{self._account} is not registered on signal-cli; run `assistai signal link`"
                )
        log.info("signal.ready", mode=mode or "unknown")

    async def _health(self) -> None:
        response = await self._get(f"{self._base}/v1/health")
        if response.status_code not in {200, 204}:
            raise SignalUnavailableError(f"signal-cli health HTTP {response.status_code}")

    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET, translating transport failures into SignalUnavailableError."""
        try:
            return await self._http.get(url, **kwargs)
        except httpx.HTTPError as exc:
            raise SignalUnavailableError("signal-cli is unreachable") from exc

    async def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        """POST, translating transport failures into SignalUnavailableError."""
        try:
            return await self._http.post(url, **kwargs)
        except httpx.HTTPError as exc:
            raise SignalUnavailableError("signal-cli is unreachable") from exc

    async def send(self, recipient: str, text: str) -> None:
        """POST /v2/send. Empty text is skipped; long text is split."""
        if self._account is None:
            raise SignalError("ASSISTAI_SIGNAL_ACCOUNT is not set")
        body = text.strip()
        if not body:
            return
        for part in _chunks(body, _MAX_SEND_CHARS):
            await self._send_part(recipient, part)

    async def receive(self, stop: asyncio.Event) -> AsyncIterator[object]:
        """Yield decoded envelopes until ``stop`` is set. Reconnects on drop."""
        if self._account is None:
            raise SignalError("ASSISTAI_SIGNAL_ACCOUNT is not set")
        delay = self._settings.signal_reconnect_seconds
        while not stop.is_set():
            try:
                async with self._connect(self.receive_url()) as conn:
                    delay = self._settings.signal_reconnect_seconds
                    log.info("signal.receive_connected")
                    closer = _close_on_stop(stop, conn)
                    try:
                        async for raw in conn:
                            decoded = _decode_frame(raw)
                            if decoded is not None:
                                yield decoded
                    finally:
                        closer.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await closer
            except Exception as exc:
                if stop.is_set():
                    return
                log.warning(
                    "signal.receive_disconnected",
                    error=type(exc).__name__,
                    reconnect_seconds=delay,
                )
                with suppress(TimeoutError):
                    await _wait(stop, delay)
                delay = min(delay * 2, 30.0)

    def receive_url(self) -> str:
        """WebSocket URL for json-rpc receive. ``+`` is percent-encoded."""
        if self._account is None:
            raise SignalError("ASSISTAI_SIGNAL_ACCOUNT is not set")
        parsed = urlparse(self._base)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = f"/v1/receive/{quote(self._account, safe='')}"
        query = "ignore_stories=true&ignore_attachments=true"
        return urlunparse((scheme, parsed.netloc, path, "", query, ""))

    async def link_uri(self, device_name: str) -> str:
        """GET /v1/qrcodelink/raw — a URI the operator scans from Signal."""
        response = await self._get(
            f"{self._base}/v1/qrcodelink/raw",
            params={"device_name": device_name},
        )
        if response.status_code >= 400:
            raise SignalError(_http_error(response))
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise SignalError("qrcodelink/raw did not return JSON") from exc
        uri = payload.get("device_link_uri") if isinstance(payload, dict) else None
        if not isinstance(uri, str) or not uri:
            raise SignalError("qrcodelink/raw missing device_link_uri")
        return uri

    async def register(self, number: str, *, captcha: str | None, voice: bool) -> None:
        body: dict[str, object] = {}
        if captcha:
            body["captcha"] = captcha
        if voice:
            body["use_voice"] = True
        response = await self._post(f"{self._base}/v1/register/{quote(number, safe='')}", json=body)
        if response.status_code >= 400:
            raise SignalError(_http_error(response))

    async def verify(self, number: str, token: str) -> None:
        response = await self._post(
            f"{self._base}/v1/register/{quote(number, safe='')}/verify/{quote(token, safe='')}"
        )
        if response.status_code >= 400:
            raise SignalError(_http_error(response))

    async def accounts(self) -> list[str]:
        listed = await self._list_accounts()
        if listed is None:
            raise SignalError("could not list Signal accounts")
        return listed

    async def _list_accounts(self) -> list[str] | None:
        try:
            response = await self._http.get(f"{self._base}/v1/accounts")
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, list):
            return None
        return [item for item in payload if isinstance(item, str)]

    async def _send_part(self, recipient: str, text: str) -> None:
        payload = {
            "number": self._account,
            "recipients": [recipient],
            "message": text,
        }
        try:
            response = await self._http.post(f"{self._base}/v2/send", json=payload)
        except httpx.HTTPError as exc:
            raise SignalError("signal-cli send failed") from exc
        if response.status_code >= 400:
            raise SignalError(_http_error(response))


def _default_connect(url: str) -> AbstractAsyncContextManager[WebSocketConnection]:
    return ws_connect(url)


def _chunks(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _decode_frame(raw: str | bytes) -> object | None:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode()
        except UnicodeDecodeError:
            log.warning("signal.frame_not_utf8")
            return None
    if not raw.strip():
        return None
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("signal.frame_not_json")
        return None
    return parsed


def _about_mode(response: httpx.Response) -> str | None:
    if response.status_code >= 400:
        return None
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    mode = payload.get("mode")
    return mode if isinstance(mode, str) else None


def _http_error(response: httpx.Response) -> str:
    body = response.text[:200].replace("\n", " ")
    return f"signal-cli HTTP {response.status_code}: {body}"


def _close_on_stop(stop: asyncio.Event, conn: WebSocketConnection) -> asyncio.Task[None]:
    async def _run() -> None:
        await stop.wait()
        with suppress(Exception):
            await conn.close()

    return asyncio.create_task(_run())


async def _wait(stop: asyncio.Event, seconds: float) -> None:
    await asyncio.wait_for(stop.wait(), timeout=seconds)
