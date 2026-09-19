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
from assistai.signal.numbers import InvalidNumberError, normalize_e164

log = structlog.get_logger(__name__)

# Older Signal clients truncate around 2 KiB. Stay under that and send parts.
MAX_SEND_CHARS = 1900


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

    async def number_for_uuid(self, uuid: str) -> str | None: ...


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
        for part in _chunks(body, MAX_SEND_CHARS):
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
                            size = _raw_size(raw)
                            if size > self._settings.signal_max_receive_bytes:
                                log.warning(
                                    "signal.frame_too_large",
                                    bytes=size,
                                    limit=self._settings.signal_max_receive_bytes,
                                )
                                continue
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

    async def number_for_uuid(self, uuid: str) -> str | None:
        """Resolve a Signal ACI to E.164 when the contact store knows it.

        Phone-number privacy often splits one person into a UUID-only row and a
        number-only row. If that UUID is known and exactly one other contact
        has a number and no UUID (besides this account), pair them.
        """
        if self._account is None:
            return None
        try:
            response = await self._get(f"{self._base}/v1/contacts/{quote(self._account, safe='')}")
        except SignalUnavailableError:
            return None
        if response.status_code >= 400:
            return None
        try:
            payload: object = response.json()
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, list):
            return None
        want = uuid.strip().lower()
        number_only: list[str] = []
        uuid_without_number = False
        for item in payload:
            if not isinstance(item, dict):
                continue
            listed_uuid = item.get("uuid")
            listed_uuid = (
                listed_uuid.strip().lower()
                if isinstance(listed_uuid, str) and listed_uuid.strip()
                else ""
            )
            listed_number = _contact_number(item.get("number"))
            if listed_uuid == want:
                if listed_number is not None:
                    return listed_number
                uuid_without_number = True
                continue
            if listed_number is not None and not listed_uuid and listed_number != self._account:
                number_only.append(listed_number)
        if uuid_without_number and len(number_only) == 1:
            return number_only[0]
        return None

    async def lift_rate_limit(self, *, challenge_token: str, captcha: str) -> None:
        """POST /v1/accounts/{number}/rate-limit-challenge after a 429 send."""
        if self._account is None:
            raise SignalError("ASSISTAI_SIGNAL_ACCOUNT is not set")
        token = challenge_token.strip()
        proof = captcha.strip()
        if not token:
            raise SignalError("challenge_token is empty")
        if not proof.startswith("signalcaptcha://"):
            raise SignalError("captcha must start with signalcaptcha://")
        response = await self._post(
            f"{self._base}/v1/accounts/{quote(self._account, safe='')}/rate-limit-challenge",
            json={"challenge_token": token, "captcha": proof},
        )
        if response.status_code >= 400:
            raise SignalError(_http_error(response))

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
        if response.status_code == 429:
            tokens = _challenge_tokens(response)
            log.error("signal.send_rate_limited", tokens=tokens)
            suffix = f" tokens={tokens}" if tokens else ""
            raise SignalError(f"signal-cli HTTP 429: send rate-limited.{suffix}")
        if response.status_code >= 400:
            raise SignalError(_http_error(response))


def _default_connect(url: str) -> AbstractAsyncContextManager[WebSocketConnection]:
    return ws_connect(url)


def _raw_size(raw: str | bytes) -> int:
    return len(raw) if isinstance(raw, bytes) else len(raw.encode())


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


def _challenge_tokens(response: httpx.Response) -> list[str]:
    try:
        payload: object = response.json()
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    raw = payload.get("challenge_tokens")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str) and item.strip()]
    token = payload.get("challenge_token")
    if isinstance(token, str) and token.strip():
        return [token.strip()]
    return []


def _contact_number(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return normalize_e164(raw)
    except InvalidNumberError:
        return None


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
