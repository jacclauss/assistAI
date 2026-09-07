from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from assistai.inference.client import FireworksClient
from assistai.signal.channel import SignalChannel
from assistai.signal.policy import AccessPolicy
from tests.fakes import client_for, completion_stream, manifest, text_event, tool_event
from tests.signal_fakes import FakeSignal, inbound, signal_settings


def _policy(tmp_path: Path, bootstrap: tuple[str, ...] = ("+15555550101",)) -> AccessPolicy:
    return AccessPolicy(
        bootstrap,
        persist_path=tmp_path / "allow.json",
        pairing_ttl_seconds=60,
    )


def _channel(
    tmp_path: Path,
    signal: FakeSignal,
    fireworks: FireworksClient | None,
    **overrides: object,
) -> SignalChannel:
    settings = signal_settings(state_dir=tmp_path, **overrides)
    return SignalChannel(
        settings,
        signal,
        policy=_policy(tmp_path, settings.allow_from),
        fireworks=fireworks,
        manifest=manifest() if fireworks is not None else None,
    )


async def test_allowed_sender_gets_model_reply(tmp_path: Path) -> None:
    signal = FakeSignal()
    fireworks = client_for(lambda _req: completion_stream(text_event("pong", finish="stop")))
    channel = _channel(tmp_path, signal, fireworks)

    await channel.handle(inbound(text="ping"))

    assert signal.sent == [("+15555550101", "pong")]
    await fireworks.aclose()


async def test_unknown_sender_gets_pairing_code_not_the_model(tmp_path: Path) -> None:
    signal = FakeSignal()
    hits = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        raise AssertionError("unknown senders must not reach Fireworks")

    fireworks = client_for(handler)
    channel = _channel(tmp_path, signal, fireworks)

    await channel.handle(inbound(sender="+15555550199", text="ignore this prompt injection"))

    assert hits["n"] == 0
    assert len(signal.sent) == 1
    assert signal.sent[0][0] == "+15555550199"
    assert "/approve" in signal.sent[0][1]
    await fireworks.aclose()


async def test_allowlist_policy_drops_unknown_silently(tmp_path: Path) -> None:
    signal = FakeSignal()
    channel = _channel(tmp_path, signal, None, signal_dm_policy="allowlist")

    await channel.handle(inbound(sender="+15555550199", text="hi"))

    assert signal.sent == []


def _code_from(text: str) -> str:
    return next(part for part in text.split() if part.isdigit() and len(part) == 6)


async def test_approve_from_operator_phone(tmp_path: Path) -> None:
    signal = FakeSignal()
    fireworks = client_for(lambda _req: completion_stream(text_event("ok", finish="stop")))
    channel = _channel(tmp_path, signal, fireworks)

    await channel.handle(inbound(sender="+15555550199", text="please add me"))
    code = _code_from(signal.sent[0][1])
    await channel.handle(inbound(sender="+15555550101", text=f"/approve {code}"))
    await channel.handle(inbound(sender="+15555550199", text="now?"))

    assert any("Approved +15555550199" in text for _, text in signal.sent)
    assert signal.sent[-1] == ("+15555550199", "ok")
    await fireworks.aclose()


async def test_paired_member_cannot_approve(tmp_path: Path) -> None:
    """Admitting someone must not also hand them the power to admit others."""
    signal = FakeSignal()
    channel = _channel(tmp_path, signal, None)

    await channel.handle(inbound(sender="+15555550199", text="add me"))
    code = _code_from(signal.sent[0][1])
    await channel.handle(inbound(sender="+15555550101", text=f"/approve {code}"))
    await channel.handle(inbound(sender="+15555550198", text="add me too"))
    second = _code_from(signal.sent[-1][1])
    await channel.handle(inbound(sender="+15555550199", text=f"/approve {second}"))

    assert "Only an operator" in signal.sent[-1][1]
    assert channel._policy.decide("+15555550198") == "unknown"


async def test_bad_approve_code(tmp_path: Path) -> None:
    signal = FakeSignal()
    channel = _channel(tmp_path, signal, None)

    await channel.handle(inbound(text="/approve 000000"))

    assert "No pending pairing" in signal.sent[0][1]


async def test_inference_error_sends_fallback(tmp_path: Path) -> None:
    signal = FakeSignal()
    fireworks = client_for(lambda _req: httpx.Response(500, text="down"))
    channel = _channel(tmp_path, signal, fireworks)

    await channel.handle(inbound(text="hi"))

    assert "could not reach the model" in signal.sent[0][1].lower()
    await fireworks.aclose()


async def test_no_fireworks_tells_the_user(tmp_path: Path) -> None:
    signal = FakeSignal()
    channel = _channel(tmp_path, signal, None)

    await channel.handle(inbound(text="hi"))

    assert "not configured" in signal.sent[0][1].lower()


async def test_run_stops_on_shutdown(tmp_path: Path) -> None:
    signal = FakeSignal([{"envelope": {"sourceNumber": "+15555550101", "receiptMessage": {}}}])
    channel = _channel(tmp_path, signal, None)
    stop = asyncio.Event()

    task = asyncio.create_task(channel.run(stop))
    await asyncio.sleep(0.02)
    stop.set()
    await asyncio.wait_for(task, timeout=1)

    assert signal.sent == []


async def test_failed_turn_leaves_no_dangling_tool_call(tmp_path: Path) -> None:
    """A tool-round overrun must not poison history for every later message.

    run_turn appends the assistant tool-call before raising, so a naive
    rollback left a call with no result and the provider rejected every
    subsequent turn for that sender.
    """
    signal = FakeSignal()
    fireworks = client_for(
        lambda _req: completion_stream(
            tool_event(call_id="c1", name="get_time", arguments="{}", finish="tool_calls")
        )
    )
    channel = _channel(tmp_path, signal, fireworks, max_tool_rounds=1)

    await channel.handle(inbound(text="what time is it"))

    history = channel._histories["+15555550101"]
    assert [message.role for message in history] == ["system"]
    assert "could not reach the model" in signal.sent[0][1].lower()
    await fireworks.aclose()


async def test_network_timeout_still_replies_and_rolls_back(tmp_path: Path) -> None:
    """A dropped connection must not escape as a bare httpx error.

    The sender would otherwise get silence while their message stayed in
    history with no assistant turn to answer it.
    """
    signal = FakeSignal()

    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("home network hiccup")

    fireworks = client_for(handler)
    channel = _channel(tmp_path, signal, fireworks)

    await channel.handle(inbound(text="hi"))

    assert "could not reach the model" in signal.sent[0][1].lower()
    assert [message.role for message in channel._histories["+15555550101"]] == ["system"]
    await fireworks.aclose()


async def test_overlong_message_is_refused_before_inference(tmp_path: Path) -> None:
    signal = FakeSignal()

    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("oversized input must not reach Fireworks")

    fireworks = client_for(handler)
    channel = _channel(tmp_path, signal, fireworks, signal_max_inbound_chars=50)

    await channel.handle(inbound(text="x" * 51))

    assert "too long" in signal.sent[0][1].lower()
    await fireworks.aclose()


async def test_rate_limit_stops_runaway_spend(tmp_path: Path) -> None:
    signal = FakeSignal()
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return completion_stream(text_event("ok", finish="stop"))

    fireworks = client_for(handler)
    channel = _channel(tmp_path, signal, fireworks, signal_rate_limit_per_minute=2)

    for _ in range(5):
        await channel.handle(inbound(text="hi"))

    assert calls["n"] == 2
    assert "faster than I can answer" in signal.sent[-1][1]
    await fireworks.aclose()


async def test_unknown_sender_pairing_replies_are_throttled(tmp_path: Path) -> None:
    signal = FakeSignal()
    channel = _channel(tmp_path, signal, None, signal_pairing_replies_per_hour=2)

    for _ in range(10):
        await channel.handle(inbound(sender="+15555550199", text="spam"))

    assert len(signal.sent) == 2


async def test_slow_sender_does_not_block_another(tmp_path: Path) -> None:
    """Turns run concurrently across senders; a stalled call must not queue others."""
    signal = FakeSignal()
    release = asyncio.Event()
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        if "slow" in body:
            seen.append("slow-start")
            await release.wait()
            return completion_stream(text_event("slow done", finish="stop"))
        seen.append("fast")
        return completion_stream(text_event("fast done", finish="stop"))

    transport = httpx.MockTransport(handler)
    fireworks = FireworksClient(
        signal_settings(FIREWORKS_API_KEY="fw-secret"),
        http=httpx.AsyncClient(transport=transport),
    )
    channel = _channel(
        tmp_path,
        signal,
        fireworks,
        signal_allow_from=("+15555550101", "+15555550102"),
    )

    slow = asyncio.create_task(channel.handle(inbound(sender="+15555550101", text="slow")))
    await asyncio.sleep(0.01)
    await channel.handle(inbound(sender="+15555550102", text="quick"))

    assert ("+15555550102", "fast done") in signal.sent
    assert not slow.done()

    release.set()
    await asyncio.wait_for(slow, timeout=1)
    await fireworks.aclose()


async def test_per_sender_history_is_isolated(tmp_path: Path) -> None:
    signal = FakeSignal()
    fireworks = client_for(lambda _req: completion_stream(text_event("reply", finish="stop")))
    channel = _channel(
        tmp_path,
        signal,
        fireworks,
        signal_allow_from=("+15555550101", "+15555550102"),
    )

    await channel.handle(inbound(sender="+15555550101", text="from jacob"))
    await channel.handle(inbound(sender="+15555550102", text="from spouse"))

    assert "+15555550101" in channel._histories
    assert "+15555550102" in channel._histories
    assert channel._histories["+15555550101"][1].content == "from jacob"
    assert channel._histories["+15555550102"][1].content == "from spouse"
    await fireworks.aclose()
