from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import SecretStr

from assistai.config import Settings
from assistai.errors import SignalUnavailableError
from assistai.gateway import Gateway
from tests.fakes import PRIMARY, QUARANTINE, catalog_response, client_for, manifest
from tests.signal_fakes import FakeChannel, FakeSignal, signal_settings


async def test_heartbeat_emits_until_shutdown() -> None:
    gateway = Gateway(Settings(heartbeat_seconds=0.01))

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.05)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert gateway.beats >= 2


async def test_shutdown_does_not_wait_out_the_interval() -> None:
    """SIGTERM must not block for the heartbeat interval before the process exits."""
    gateway = Gateway(Settings(heartbeat_seconds=3600))

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.01)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert gateway.beats == 1


async def test_repeated_shutdown_requests_are_harmless() -> None:
    gateway = Gateway(Settings(heartbeat_seconds=0.01))

    task = asyncio.create_task(gateway.run())
    gateway.request_shutdown("first")
    gateway.request_shutdown("second")
    await asyncio.wait_for(task, timeout=1)

    assert task.done()


async def test_skips_validation_without_api_key() -> None:
    gateway = Gateway(Settings(heartbeat_seconds=0.01))

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.02)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert gateway.validated_refs == ()


async def test_validates_pinned_models_when_key_present() -> None:
    client = client_for(lambda _req: catalog_response([PRIMARY, QUARANTINE]))
    gateway = Gateway(
        Settings(FIREWORKS_API_KEY=SecretStr("fw-secret"), heartbeat_seconds=0.01),
        client=client,
        manifest=manifest(),
    )

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.02)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert gateway.validated_refs == (PRIMARY, QUARANTINE)
    await client.aclose()


async def test_skips_signal_without_account() -> None:
    gateway = Gateway(Settings(heartbeat_seconds=0.01))

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.02)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert gateway._channel_task is None


async def test_starts_injected_signal_channel() -> None:
    channel = FakeChannel()
    gateway = Gateway(
        signal_settings(heartbeat_seconds=0.01, FIREWORKS_API_KEY=None),
        signal=FakeSignal(),
        channel=channel,
    )

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.02)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert channel.ran is True


async def test_builds_a_real_channel_and_waits_for_signal_cli(tmp_path: Path) -> None:
    """Covers the wiring _start_signal does when no channel is injected."""
    signal = FakeSignal()
    gateway = Gateway(
        signal_settings(
            heartbeat_seconds=0.01,
            FIREWORKS_API_KEY=None,
            state_dir=tmp_path,
        ),
        signal=signal,
    )

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.02)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert signal.waited is True
    assert signal.check_calls == 1


async def test_unreachable_signal_cli_fails_startup() -> None:
    class Dead(FakeSignal):
        async def wait_until_healthy(
            self, timeout_seconds: float, stop: asyncio.Event, *, poll_seconds: float = 3.0
        ) -> None:
            raise SignalUnavailableError("down")

    gateway = Gateway(
        signal_settings(heartbeat_seconds=0.01, FIREWORKS_API_KEY=None),
        signal=Dead(),
    )

    with pytest.raises(SignalUnavailableError):
        await gateway.run()
