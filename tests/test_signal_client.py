from __future__ import annotations

import asyncio
import json
from contextlib import AbstractAsyncContextManager

import httpx
import pytest

from assistai.errors import SignalError, SignalUnavailableError
from assistai.signal.client import SignalClient, WebSocketConnection
from tests.signal_fakes import http_signal, scripted_ws, signal_settings


def _about(mode: str = "json-rpc") -> httpx.Response:
    return httpx.Response(200, json={"mode": mode, "versions": ["v1", "v2"]})


def test_receive_url_encodes_plus() -> None:
    client = SignalClient(signal_settings(), http=httpx.AsyncClient())

    url = client.receive_url()

    assert url.startswith("ws://signal.test/v1/receive/%2B15555550100")
    assert "ignore_stories=true" in url


def test_https_base_uses_wss() -> None:
    client = SignalClient(
        signal_settings(signal_base_url="https://signal.example"),
        http=httpx.AsyncClient(),
    )

    assert client.receive_url().startswith("wss://")


async def test_check_accepts_204_and_warns_on_wrong_mode() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/health":
            return httpx.Response(204)
        if request.url.path == "/v1/about":
            return _about("normal")
        if request.url.path == "/v1/accounts":
            return httpx.Response(200, json=["+15555550100"])
        raise AssertionError(request.url)

    client = http_signal(handler)
    await client.check()
    await client.aclose()


async def test_check_fails_when_unreachable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = http_signal(handler)
    with pytest.raises(SignalUnavailableError):
        await client.check()
    await client.aclose()


async def test_link_uri_transport_failure_is_a_signal_error() -> None:
    """`assistai signal link` should print a message, not a traceback."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = http_signal(handler)
    with pytest.raises(SignalUnavailableError):
        await client.link_uri("assistai")
    await client.aclose()


async def test_register_transport_failure_is_a_signal_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = http_signal(handler)
    with pytest.raises(SignalUnavailableError):
        await client.register("+15555550100", captcha=None, voice=False)
    await client.aclose()


async def test_check_survives_about_endpoint_dropping() -> None:
    """Health can pass and /v1/about still fail mid-startup."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/health":
            return httpx.Response(204)
        raise httpx.ReadTimeout("stalled")

    client = http_signal(handler)
    with pytest.raises(SignalUnavailableError):
        await client.check()
    await client.aclose()


async def test_check_fails_when_account_not_registered() -> None:
    """Otherwise the receive socket reconnect-loops forever against nothing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/health":
            return httpx.Response(204)
        if request.url.path == "/v1/about":
            return _about()
        if request.url.path == "/v1/accounts":
            return httpx.Response(200, json=[])
        raise AssertionError(request.url)

    client = http_signal(handler)
    with pytest.raises(SignalUnavailableError, match="not registered"):
        await client.check()
    await client.aclose()


async def test_wait_until_healthy_retries_a_cold_signal_cli() -> None:
    """signal-cli boots slowly; the gateway waits instead of exiting."""
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("not up yet")
        return httpx.Response(204)

    client = http_signal(handler)
    await client.wait_until_healthy(30.0, asyncio.Event(), poll_seconds=0.01)

    assert attempts["n"] == 3
    await client.aclose()


async def test_wait_until_healthy_gives_up_at_the_deadline() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = http_signal(handler)
    with pytest.raises(SignalUnavailableError):
        await client.wait_until_healthy(0.0, asyncio.Event())
    await client.aclose()


async def test_wait_until_healthy_aborts_on_shutdown() -> None:
    stop = asyncio.Event()
    stop.set()

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = http_signal(handler)
    with pytest.raises(SignalUnavailableError):
        await client.wait_until_healthy(30.0, stop)
    await client.aclose()


async def test_send_posts_v2_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"timestamp": 1})

    client = http_signal(handler)
    await client.send("+15555550101", "hello")
    await client.aclose()

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/v2/send"
    body = json.loads(seen[0].content)
    assert body == {
        "number": "+15555550100",
        "recipients": ["+15555550101"],
        "message": "hello",
    }


async def test_send_splits_long_text() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["message"])
        return httpx.Response(201, json={"timestamp": 1})

    client = http_signal(handler)
    await client.send("+15555550101", "x" * 2000)
    await client.aclose()

    assert len(seen) == 2
    assert seen[0] == "x" * 1900
    assert seen[1] == "x" * 100


async def test_send_skips_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not send")

    client = http_signal(handler)
    await client.send("+15555550101", "   ")
    await client.aclose()


async def test_send_error_does_not_echo_body_secrets() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad token fw_secret")

    client = http_signal(handler)
    with pytest.raises(SignalError, match="HTTP 400"):
        await client.send("+15555550101", "hi")
    await client.aclose()


async def test_receive_yields_decoded_frames_then_stops() -> None:
    frames = [json.dumps({"envelope": {"sourceNumber": "+15555550101"}})]

    def connect(_url: str) -> AbstractAsyncContextManager[WebSocketConnection]:
        return scripted_ws(frames)

    client = SignalClient(signal_settings(), http=httpx.AsyncClient(), connect=connect)
    stop = asyncio.Event()
    got: list[object] = []
    async for payload in client.receive(stop):
        got.append(payload)
        stop.set()

    assert got == [{"envelope": {"sourceNumber": "+15555550101"}}]
    await client.aclose()


async def test_receive_skips_malformed_frames() -> None:
    def connect(_url: str) -> AbstractAsyncContextManager[WebSocketConnection]:
        return scripted_ws(["not-json", json.dumps({"ok": True})])

    client = SignalClient(signal_settings(), http=httpx.AsyncClient(), connect=connect)
    stop = asyncio.Event()
    got: list[object] = []
    async for payload in client.receive(stop):
        got.append(payload)
        stop.set()

    assert got == [{"ok": True}]
    await client.aclose()


async def test_link_uri_reads_device_link() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/qrcodelink/raw"
        return httpx.Response(200, json={"device_link_uri": "sgnl://link?uuid=abc"})

    client = http_signal(handler)
    uri = await client.link_uri("assistai")
    assert uri == "sgnl://link?uuid=abc"
    await client.aclose()


async def test_register_and_verify_paths() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(201)

    client = http_signal(handler)
    await client.register("+15555550100", captcha="token", voice=True)
    await client.verify("+15555550100", "123-456")
    await client.aclose()

    assert seen[0].startswith("POST /v1/register/")
    assert "15555550100" in seen[0]
    assert "verify" in seen[1]
    assert "123-456" in seen[1]
