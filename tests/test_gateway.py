from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import pytest
from pydantic import SecretStr
from structlog.testing import capture_logs

from assistai.config import Settings
from assistai.errors import HouseholdConfigError, SignalUnavailableError
from assistai.gateway import Gateway, install_signal_handlers
from tests.fakes import PRIMARY, QUARANTINE, catalog_response, client_for, manifest
from tests.signal_fakes import FakeChannel, FakeSignal, signal_settings


def _agents_toml(tmp_path: Path) -> Path:
    path = tmp_path / "assistai.toml"
    path.write_text(
        """
[agents.jacob]
binds_to = { channel = "signal", peer = "+15555550101" }
tools = []
[agents.spouse]
binds_to = { channel = "signal", peer = "+15555550102" }
tools = []
""",
        encoding="utf-8",
    )
    return path


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
            agents_config_path=_agents_toml(tmp_path),
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


async def test_missing_agent_config_fails_signal_startup(tmp_path: Path) -> None:
    gateway = Gateway(
        signal_settings(
            heartbeat_seconds=0.01,
            FIREWORKS_API_KEY=None,
            state_dir=tmp_path,
            agents_config_path=tmp_path / "missing.toml",
        ),
        signal=FakeSignal(),
    )

    with pytest.raises(HouseholdConfigError):
        await gateway.run()


async def test_a_crashing_channel_shuts_the_gateway_down() -> None:
    """Otherwise the process lives on as a heartbeat that answers nobody.

    A dead channel that keeps the container 'healthy' is worse than exiting:
    the runtime would restart a broken gateway if it knew, and it cannot.
    """

    class Crashing:
        async def run(self, _stop: asyncio.Event) -> None:
            raise RuntimeError("receive loop exploded")

    gateway = Gateway(
        signal_settings(heartbeat_seconds=3600, FIREWORKS_API_KEY=None),
        signal=FakeSignal(),
        channel=Crashing(),
    )

    await asyncio.wait_for(gateway.run(), timeout=1)


async def test_a_hung_channel_is_cancelled_at_shutdown() -> None:
    """SIGTERM has a grace period. Waiting past it means being killed."""

    class Hung:
        def __init__(self) -> None:
            self.cancelled = False

        async def run(self, _stop: asyncio.Event) -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    channel = Hung()
    gateway = Gateway(
        signal_settings(
            heartbeat_seconds=0.01,
            FIREWORKS_API_KEY=None,
            shutdown_grace_seconds=0.05,
        ),
        signal=FakeSignal(),
        channel=channel,
    )

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.02)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert channel.cancelled is True


async def test_owned_clients_are_closed_on_exit(tmp_path: Path) -> None:
    """A leaked signal-cli socket blocks the account on the next start."""
    signal = FakeSignal()
    closed = {"n": 0}

    async def aclose() -> None:
        closed["n"] += 1

    signal.aclose = aclose  # type: ignore[method-assign]
    gateway = Gateway(
        signal_settings(
            heartbeat_seconds=0.01,
            FIREWORKS_API_KEY=None,
            state_dir=tmp_path,
            agents_config_path=_agents_toml(tmp_path),
        ),
    )
    gateway._signal = signal
    gateway._owns_signal = True

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.02)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert closed["n"] == 1


async def test_allowlisted_number_with_no_agent_is_flagged_at_startup(
    tmp_path: Path,
) -> None:
    """The operator's only warning that a number will reach nothing."""
    gateway = Gateway(
        signal_settings(
            heartbeat_seconds=0.01,
            FIREWORKS_API_KEY=None,
            state_dir=tmp_path,
            agents_config_path=_agents_toml(tmp_path),
            signal_allow_from=("+15555550101", "+15555550199"),
        ),
        signal=FakeSignal(),
    )

    with capture_logs() as logs:
        task = asyncio.create_task(gateway.run())
        await asyncio.sleep(0.02)
        gateway.request_shutdown("test")
        await asyncio.wait_for(task, timeout=1)

    warned = next(entry for entry in logs if entry["event"] == "gateway.allow_from_unbound")
    assert warned["numbers"] == ["+15555550199"]


async def test_empty_allowlist_is_flagged_at_startup(tmp_path: Path) -> None:
    """With no operator, nobody can ever approve a pairing code."""
    gateway = Gateway(
        signal_settings(
            heartbeat_seconds=0.01,
            FIREWORKS_API_KEY=None,
            state_dir=tmp_path,
            agents_config_path=_agents_toml(tmp_path),
            signal_allow_from=(),
        ),
        signal=FakeSignal(),
    )

    with capture_logs() as logs:
        task = asyncio.create_task(gateway.run())
        await asyncio.sleep(0.02)
        gateway.request_shutdown("test")
        await asyncio.wait_for(task, timeout=1)

    assert any(entry["event"] == "gateway.signal_allowlist_empty" for entry in logs)


async def test_sigterm_requests_a_graceful_shutdown() -> None:
    """Container runtimes send SIGTERM, then kill. It cannot be an error path."""
    gateway = Gateway(Settings(heartbeat_seconds=3600))
    install_signal_handlers(gateway)

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.01)
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(task, timeout=1)

    assert gateway.beats == 1
