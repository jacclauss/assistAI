"""Signal doubles for unit tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any

import httpx

from assistai.config import Settings
from assistai.signal.client import SignalClient
from assistai.signal.envelopes import InboundText


def signal_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "signal_account": "+15555550100",
        "signal_allow_from": ("+15555550101",),
        "signal_base_url": "http://signal.test",
        "signal_reconnect_seconds": 0.01,
        "signal_pairing_ttl_seconds": 60,
    }
    values.update(overrides)
    return Settings(**values)


def envelope(
    *,
    sender: str = "+15555550101",
    text: str = "hello",
    group: bool = False,
    timestamp: int = 1,
) -> dict[str, Any]:
    data: dict[str, Any] = {"message": text, "timestamp": timestamp}
    if group:
        data["groupInfo"] = {"groupId": "abc", "revision": 1}
    return {
        "envelope": {
            "sourceNumber": sender,
            "timestamp": timestamp,
            "dataMessage": data,
        }
    }


def inbound(sender: str = "+15555550101", text: str = "hello") -> InboundText:
    return InboundText(sender=sender, text=text, timestamp=1)


class FakeSignal:
    """In-memory SignalClient stand-in for the channel and gateway."""

    def __init__(self, frames: Iterable[object] | None = None) -> None:
        self.inbox: asyncio.Queue[object] = asyncio.Queue()
        self.sent: list[tuple[str, str]] = []
        self.check_calls = 0
        self.waited = False
        for frame in frames or ():
            self.inbox.put_nowait(frame)

    async def check(self) -> None:
        self.check_calls += 1

    async def wait_until_healthy(
        self, timeout_seconds: float, stop: asyncio.Event, *, poll_seconds: float = 3.0
    ) -> None:
        self.waited = True

    async def send(self, recipient: str, text: str) -> None:
        self.sent.append((recipient, text))

    async def receive(self, stop: asyncio.Event) -> AsyncIterator[object]:
        while not stop.is_set():
            get = asyncio.create_task(self.inbox.get())
            wait = asyncio.create_task(stop.wait())
            done, pending = await asyncio.wait({get, wait}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if wait in done:
                return
            yield get.result()

    async def aclose(self) -> None:
        return None


class FakeChannel:
    def __init__(self) -> None:
        self.ran = False

    async def run(self, stop: asyncio.Event) -> None:
        self.ran = True
        await stop.wait()


def http_signal(handler: httpx.MockTransport | Any, **overrides: Any) -> SignalClient:
    transport = (
        handler if isinstance(handler, httpx.MockTransport) else httpx.MockTransport(handler)
    )
    http = httpx.AsyncClient(transport=transport)
    return SignalClient(signal_settings(**overrides), http=http)


class ScriptedSocket:
    def __init__(self, frames: Iterable[str | bytes]) -> None:
        self._frames = list(frames)
        self.closed = False

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str | bytes]:
        for frame in self._frames:
            yield frame

    async def close(self) -> None:
        self.closed = True


@asynccontextmanager
async def scripted_ws(frames: Iterable[str | bytes]) -> AsyncIterator[ScriptedSocket]:
    yield ScriptedSocket(frames)


class BlockingSocket:
    """Yields its frames, then hangs like a real idle WebSocket.

    A socket that ends on its own cannot show whether shutdown unblocks the
    receive loop, which is the case that decides if the gateway can stop.
    """

    def __init__(self, frames: Iterable[str | bytes]) -> None:
        self._frames = list(frames)
        self._closed = asyncio.Event()
        self.closed = False

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str | bytes]:
        for frame in self._frames:
            yield frame
        await self._closed.wait()

    async def close(self) -> None:
        self.closed = True
        self._closed.set()
