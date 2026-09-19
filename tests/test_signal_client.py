from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx
import pytest

from assistai.errors import SignalError, SignalUnavailableError
from assistai.signal.client import SignalClient, WebSocketConnection
from tests.signal_fakes import BlockingSocket, http_signal, scripted_ws, signal_settings


def _about(mode: str = "json-rpc") -> httpx.Response:
    return httpx.Response(200, json={"mode": mode, "versions": ["v1", "v2"]})


def test_receive_url_encodes_plus() -> None:
    client = SignalClient(signal_settings(), http=httpx.AsyncClient())

    url = client.receive_url()

    assert url.startswith("ws://signal.test/v1/receive/%2B15555550100")
    assert "ignore_stories=true" in url
    assert "ignore_attachments=true" in url


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


async def test_number_for_uuid_reads_contacts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/v1/contacts/" in request.url.path
        assert "15555550100" in request.url.path
        return httpx.Response(
            200,
            json=[
                {
                    "number": "+15555550101",
                    "uuid": "429cce0e-9174-4d7a-a98b-1cb9208b1951",
                }
            ],
        )

    client = http_signal(handler)
    found = await client.number_for_uuid("429CCE0E-9174-4D7A-A98B-1CB9208B1951")
    await client.aclose()
    assert found == "+15555550101"


async def test_number_for_uuid_stitches_split_contact() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"number": "+15555550101", "uuid": ""},
                {"number": "", "uuid": "429cce0e-9174-4d7a-a98b-1cb9208b1951"},
                {"number": "+15555550100", "uuid": "13a428c0-42fb-4e89-ad6c-f50c8e7900a7"},
            ],
        )

    client = http_signal(handler)
    found = await client.number_for_uuid("429cce0e-9174-4d7a-a98b-1cb9208b1951")
    await client.aclose()
    assert found == "+15555550101"


async def test_number_for_uuid_returns_none_when_contacts_fail() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = http_signal(handler)
    found = await client.number_for_uuid("429cce0e-9174-4d7a-a98b-1cb9208b1951")
    await client.aclose()
    assert found is None


async def test_send_skips_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not send")

    client = http_signal(handler)
    await client.send("+15555550101", "   ")
    await client.aclose()


async def test_send_rate_limit_includes_challenge_tokens() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": "rate limited",
                "challenge_tokens": ["3472e52f-7416-4e1a-8da3-668dfb59557c"],
            },
        )

    client = http_signal(handler)
    with pytest.raises(SignalError, match="3472e52f-7416-4e1a-8da3-668dfb59557c"):
        await client.send("+15555550101", "hello")
    await client.aclose()


async def test_lift_rate_limit_posts_challenge() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    client = http_signal(handler)
    await client.lift_rate_limit(
        challenge_token="3472e52f-7416-4e1a-8da3-668dfb59557c",
        captcha="signalcaptcha://proof",
    )
    await client.aclose()

    assert seen[0].method == "POST"
    assert "/rate-limit-challenge" in seen[0].url.path
    assert json.loads(seen[0].content) == {
        "challenge_token": "3472e52f-7416-4e1a-8da3-668dfb59557c",
        "captcha": "signalcaptcha://proof",
    }


async def test_lift_rate_limit_rejects_non_captcha() -> None:
    client = http_signal(lambda _request: httpx.Response(204))
    with pytest.raises(SignalError, match="signalcaptcha://"):
        await client.lift_rate_limit(challenge_token="abc", captcha="not-a-captcha")
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


async def test_receive_drops_oversized_frames() -> None:
    huge = json.dumps({"envelope": {"sourceNumber": "+15555550101", "pad": "x" * 300_000}})

    def connect(_url: str) -> AbstractAsyncContextManager[WebSocketConnection]:
        return scripted_ws([huge, json.dumps({"ok": True})])

    client = SignalClient(
        signal_settings(signal_max_receive_bytes=1000),
        http=httpx.AsyncClient(),
        connect=connect,
    )
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


async def test_health_reports_a_bad_status() -> None:
    client = http_signal(lambda _request: httpx.Response(500, text="boom"))

    with pytest.raises(SignalUnavailableError, match="HTTP 500"):
        await client.check()
    await client.aclose()


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(httpx.Response(409, text="already linked"), id="http-error"),
        pytest.param(httpx.Response(200, text="not json"), id="not-json"),
        pytest.param(httpx.Response(200, json={"other": "field"}), id="missing-uri"),
        pytest.param(httpx.Response(200, json={"device_link_uri": ""}), id="blank-uri"),
    ],
)
async def test_link_uri_rejects_unusable_responses(response: httpx.Response) -> None:
    """A silent empty string here would send the operator to scan nothing."""
    client = http_signal(lambda _request: response)

    with pytest.raises(SignalError):
        await client.link_uri("assistai")
    await client.aclose()


async def test_register_surfaces_a_captcha_demand() -> None:
    """The 402 body is the operator's instruction, so it has to reach them."""
    client = http_signal(
        lambda _request: httpx.Response(402, text="Captcha required for verification")
    )

    with pytest.raises(SignalError, match="Captcha required"):
        await client.register("+15555550100", captcha=None, voice=False)
    await client.aclose()


async def test_verify_surfaces_a_bad_code() -> None:
    client = http_signal(lambda _request: httpx.Response(400, text="invalid verification code"))

    with pytest.raises(SignalError, match="HTTP 400"):
        await client.verify("+15555550100", "000000")
    await client.aclose()


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(httpx.Response(500, text="boom"), id="server-error"),
        pytest.param(httpx.Response(200, text="not json"), id="not-json"),
        pytest.param(httpx.Response(200, json={"accounts": []}), id="not-a-list"),
    ],
)
async def test_accounts_fails_loudly_when_it_cannot_list(response: httpx.Response) -> None:
    """`signal health` must not print an empty roster it never actually read."""
    client = http_signal(lambda _request: response)

    with pytest.raises(SignalError, match="could not list"):
        await client.accounts()
    await client.aclose()


async def test_accounts_drops_non_string_entries() -> None:
    client = http_signal(lambda _request: httpx.Response(200, json=["+15555550100", 7, None]))

    assert await client.accounts() == ["+15555550100"]
    await client.aclose()


async def test_check_tolerates_an_accounts_endpoint_that_fails() -> None:
    """Not being able to list is not proof the account is missing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/health":
            return httpx.Response(204)
        if request.url.path == "/v1/about":
            return httpx.Response(200, text="not json")
        return httpx.Response(500)

    client = http_signal(handler)
    await client.check()
    await client.aclose()


@pytest.mark.parametrize("method", ["send", "receive_url"])
async def test_no_account_configured_is_a_clear_error(method: str) -> None:
    client = http_signal(lambda _request: httpx.Response(200), signal_account=None)

    with pytest.raises(SignalError, match="ASSISTAI_SIGNAL_ACCOUNT"):
        if method == "send":
            await client.send("+15555550101", "hi")
        else:
            client.receive_url()
    await client.aclose()


async def test_receive_without_an_account_is_a_clear_error() -> None:
    client = http_signal(lambda _request: httpx.Response(200), signal_account=None)

    with pytest.raises(SignalError, match="ASSISTAI_SIGNAL_ACCOUNT"):
        async for _ in client.receive(asyncio.Event()):
            raise AssertionError("should not yield")
    await client.aclose()


async def test_send_transport_failure_is_a_signal_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = http_signal(handler)
    with pytest.raises(SignalError, match="send failed"):
        await client.send("+15555550101", "hi")
    await client.aclose()


async def test_receive_reconnects_after_a_dropped_socket() -> None:
    """The Pi's Wi-Fi will drop. A dropped socket must not end the process.

    json-rpc mode discards anything that arrives while no client is attached,
    so a receive loop that exits on the first disconnect loses every message
    until someone notices.
    """
    attempts = {"n": 0}

    def connect(_url: str) -> AbstractAsyncContextManager[WebSocketConnection]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("connection reset")
        return scripted_ws([json.dumps({"envelope": {"sourceNumber": "+15555550101"}})])

    client = SignalClient(signal_settings(), http=httpx.AsyncClient(), connect=connect)
    stop = asyncio.Event()
    got: list[object] = []
    async for payload in client.receive(stop):
        got.append(payload)
        stop.set()

    assert attempts["n"] == 2
    assert got == [{"envelope": {"sourceNumber": "+15555550101"}}]
    await client.aclose()


async def test_receive_stops_reconnecting_once_shutdown_is_requested() -> None:
    attempts = {"n": 0}
    stop = asyncio.Event()

    def connect(_url: str) -> AbstractAsyncContextManager[WebSocketConnection]:
        attempts["n"] += 1
        stop.set()
        raise OSError("connection reset")

    client = SignalClient(signal_settings(), http=httpx.AsyncClient(), connect=connect)
    got = [payload async for payload in client.receive(stop)]

    assert got == []
    assert attempts["n"] == 1
    await client.aclose()


async def test_shutdown_unblocks_an_idle_socket() -> None:
    """An idle WebSocket never ends on its own.

    Without the stop watcher, ``receive`` would sit in ``async for`` forever
    and the gateway would hang on SIGTERM until the runtime killed it.
    """
    sockets: list[BlockingSocket] = []
    stop = asyncio.Event()

    @asynccontextmanager
    async def connect_blocking(_url: str) -> AsyncIterator[BlockingSocket]:
        socket = BlockingSocket([json.dumps({"ok": True})])
        sockets.append(socket)
        yield socket

    client = SignalClient(
        signal_settings(),
        http=httpx.AsyncClient(),
        connect=connect_blocking,
    )

    async def drain() -> list[object]:
        return [payload async for payload in client.receive(stop)]

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.02)
    assert not task.done()

    stop.set()
    got = await asyncio.wait_for(task, timeout=1)

    assert got == [{"ok": True}]
    assert sockets[0].closed is True
    await client.aclose()


async def test_receive_decodes_bytes_and_skips_junk() -> None:
    """signal-cli sends text frames, but a proxy in between may not."""
    frames: list[str | bytes] = [
        b"\xff\xfe not utf-8",
        "   ",
        json.dumps({"ok": 1}).encode(),
    ]

    def connect(_url: str) -> AbstractAsyncContextManager[WebSocketConnection]:
        return scripted_ws(frames)

    client = SignalClient(signal_settings(), http=httpx.AsyncClient(), connect=connect)
    stop = asyncio.Event()
    got: list[object] = []
    async for payload in client.receive(stop):
        got.append(payload)
        stop.set()

    assert got == [{"ok": 1}]
    await client.aclose()
