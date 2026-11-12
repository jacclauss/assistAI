from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import pytest
from pydantic import SecretStr
from structlog.testing import capture_logs

from assistai.config import Settings
from assistai.errors import AssistAIError, HouseholdConfigError, SignalUnavailableError
from assistai.gateway import Gateway, install_signal_handlers
from assistai.inference.types import Message
from assistai.jobs import Job
from assistai.scheduler import JobRunner
from assistai.store import Store, store_path
from assistai.sysclock import ClockState
from tests.clock_fakes import FakeClock, FakeClockSync, chicago
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
    """With no operator, nobody can text the bot."""
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


async def test_signal_does_not_start_without_a_timezone() -> None:
    """Without a household zone every reminder time is wrong, so nobody gets an answer."""
    channel = FakeChannel()
    gateway = Gateway(
        signal_settings(heartbeat_seconds=0.01, FIREWORKS_API_KEY=None, timezone=None),
        signal=FakeSignal(),
        channel=channel,
    )

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.05)
    gateway.request_shutdown("test")
    outcome = (await asyncio.gather(task, return_exceptions=True))[0]

    assert channel.ran is False
    # __main__ turns an AssistAIError into exit status 2.
    assert isinstance(outcome, AssistAIError)


async def test_a_missing_timezone_is_logged_for_the_operator() -> None:
    gateway = Gateway(
        signal_settings(heartbeat_seconds=0.01, FIREWORKS_API_KEY=None, timezone=None),
        signal=FakeSignal(),
        channel=FakeChannel(),
    )

    with capture_logs() as logs, pytest.raises(HouseholdConfigError, match="ASSISTAI_TIMEZONE"):
        await gateway.run()

    missing = next(entry for entry in logs if entry["event"] == "gateway.timezone_missing")
    assert "ASSISTAI_TIMEZONE" in missing["reason"]


async def test_the_zone_and_its_current_offset_are_logged_at_start() -> None:
    """The second 1:30 on fall-back day is CST. The log line says which one the gateway is in."""
    gateway = Gateway(
        signal_settings(heartbeat_seconds=0.01, FIREWORKS_API_KEY=None, timezone="America/Chicago"),
        signal=FakeSignal(),
        channel=FakeChannel(),
        now=FakeClock(chicago(2026, 11, 1, 1, 30, fold=1)),
    )

    with capture_logs() as logs:
        await _run_briefly(gateway)

    zone = next(entry for entry in logs if entry["event"] == "clock.zone")
    assert zone["zone"] == "America/Chicago"
    assert zone["local"] == "2026-11-01T01:30:00-06:00"
    assert zone["utc_offset"] == "-06:00"


async def test_signal_waits_for_a_synced_clock_then_starts() -> None:
    kernel = FakeClockSync(ClockState.UNSYNCED)
    channel = FakeChannel()
    gateway = _clock_gateway(kernel=kernel, channel=channel)

    with capture_logs() as logs:
        task = asyncio.create_task(gateway.run())
        await asyncio.sleep(0.05)
        assert channel.ran is False
        kernel.current = ClockState.SYNCED
        await asyncio.sleep(0.05)
        assert channel.ran is True
        gateway.request_shutdown("test")
        await asyncio.wait_for(task, timeout=1)

    events = [entry["event"] for entry in logs if entry["event"].startswith("clock.")]
    assert events == ["clock.zone", "clock.waiting_for_sync", "clock.synced"]
    assert kernel.asked > 2


async def test_an_unsynced_clock_starts_signal_after_the_wait() -> None:
    """A household with no bot is worse than a bot on a doubtful clock that says so."""
    channel = FakeChannel()
    gateway = _clock_gateway(
        kernel=FakeClockSync(ClockState.UNSYNCED), channel=channel, wait_seconds=0.05
    )

    with capture_logs() as logs:
        await _run_briefly(gateway, seconds=0.15)

    events = [entry["event"] for entry in logs if entry["event"].startswith("clock.")]
    assert events == ["clock.zone", "clock.waiting_for_sync", "clock.unsynced"]
    assert channel.ran is True


async def test_shutdown_during_the_clock_wait_starts_nothing() -> None:
    """Docker kills ten seconds after SIGTERM; the two-minute wait must not outlast it."""
    channel = FakeChannel()
    gateway = _clock_gateway(kernel=FakeClockSync(ClockState.UNSYNCED), channel=channel)

    with capture_logs() as logs:
        task = asyncio.create_task(gateway.run())
        await asyncio.sleep(0.03)
        gateway.request_shutdown("test")
        await asyncio.wait_for(task, timeout=0.5)

    assert channel.ran is False
    assert gateway._channel_task is None
    assert not any(entry["event"] == "gateway.signal_started" for entry in logs)


async def test_a_clock_behind_the_store_holds_startup_until_it_catches_up(
    tmp_path: Path,
) -> None:
    """With the kernel call refused, the store alone decides whether the clock is believable."""
    last_seen = 1_800_000_000.0
    _remember(tmp_path, at=last_seen)
    wall = FakeClock(last_seen - 3600)
    gateway = _clock_gateway(kernel=FakeClockSync(ClockState.UNKNOWN), tmp_path=tmp_path, now=wall)

    with capture_logs() as logs:
        task = asyncio.create_task(gateway.run())
        await asyncio.sleep(0.05)
        assert gateway._channel_task is None
        wall.set(last_seen - 120)
        await asyncio.sleep(0.05)
        assert gateway._channel_task is not None
        gateway.request_shutdown("test")
        await asyncio.wait_for(task, timeout=1)

    events = [entry["event"] for entry in logs if entry["event"].startswith("clock.")]
    assert events == ["clock.zone", "clock.sync_unknown", "clock.waiting_for_sync", "clock.synced"]
    waiting = next(entry for entry in logs if entry["event"] == "clock.waiting_for_sync")
    assert waiting["behind_store_seconds"] == 3600


async def test_a_refused_kernel_call_and_an_empty_store_start_at_once(tmp_path: Path) -> None:
    gateway = _clock_gateway(kernel=FakeClockSync(ClockState.UNKNOWN), tmp_path=tmp_path)

    with capture_logs() as logs:
        await _run_briefly(gateway)

    events = [entry["event"] for entry in logs if entry["event"].startswith("clock.")]
    assert events == ["clock.zone", "clock.sync_unknown", "clock.synced"]
    assert any(entry["event"] == "gateway.signal_started" for entry in logs)


async def test_a_future_row_does_not_hold_startup(tmp_path: Path) -> None:
    """A row stamped two days ahead is a bad row. It must not keep the bot down for good."""
    now = 1_800_000_000.0
    _remember(tmp_path, at=now + 2 * 86400)
    gateway = _clock_gateway(
        kernel=FakeClockSync(ClockState.SYNCED), tmp_path=tmp_path, now=FakeClock(now)
    )

    with capture_logs() as logs:
        await _run_briefly(gateway)

    events = [entry["event"] for entry in logs if entry["event"].startswith("clock.")]
    assert events == ["clock.zone", "clock.store_in_future", "clock.synced"]
    ahead = next(entry for entry in logs if entry["event"] == "clock.store_in_future")
    assert ahead["ahead_seconds"] == 2 * 86400
    assert any(entry["event"] == "gateway.signal_started" for entry in logs)


async def test_a_refused_kernel_call_does_not_drop_a_far_ahead_row(tmp_path: Path) -> None:
    """With no word from the kernel, a row 15 days ahead may mean the clock is 15 days behind.

    Starting Signal on that clock, at once or after the wait, saves history
    that is deleted when NTP later steps it forward past the 14-day window.
    The gateway exits instead, and the container restart waits again.
    """
    now = 1_800_000_000.0
    _remember(tmp_path, at=now + 15 * 86400)
    gateway = _clock_gateway(
        kernel=FakeClockSync(ClockState.UNKNOWN),
        tmp_path=tmp_path,
        now=FakeClock(now),
        wait_seconds=0.1,
    )

    with (
        capture_logs() as logs,
        pytest.raises(HouseholdConfigError, match="Signal is not starting"),
    ):
        await asyncio.wait_for(gateway.run(), timeout=1)

    assert gateway._channel_task is None
    assert gateway._job_task is None
    assert not any(entry["event"] == "gateway.signal_started" for entry in logs)
    events = [entry["event"] for entry in logs if entry["event"].startswith("clock.")]
    assert events == [
        "clock.zone",
        "clock.sync_unknown",
        "clock.waiting_for_sync",
        "clock.unsynced",
    ]
    waiting = next(entry for entry in logs if entry["event"] == "clock.waiting_for_sync")
    assert waiting["behind_store_seconds"] == 15 * 86400
    unsynced = next(entry for entry in logs if entry["event"] == "clock.unsynced")
    assert "Signal is not starting" in unsynced["reason"]


async def test_rows_from_an_older_build_get_a_schedule_before_the_runner_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner never reads a row with no schedule, after an upgrade or after a rollback."""
    now = 1_800_000_000.0
    store = Store(store_path(tmp_path))
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="mgmplus cancel reminder",
            prompt="Cancel MGM+ before the rate goes up.",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=now - 86400,
            expires_at=None,
            next_run_at=now + 3600,
            last_run_at=None,
            cancelled_at=None,
            verbatim=True,
        )
    )
    store.close()
    seen: list[str | None] = []

    async def runner_starts(self: JobRunner, stop: asyncio.Event) -> None:
        job = self._store.find_job("jacob", name="mgmplus cancel reminder")
        seen.append(job.zone if job is not None else "missing")
        await stop.wait()

    monkeypatch.setattr(JobRunner, "run", runner_starts)
    gateway = _clock_gateway(
        kernel=FakeClockSync(ClockState.SYNCED), tmp_path=tmp_path, now=FakeClock(now)
    )

    with capture_logs() as logs:
        await _run_briefly(gateway)

    assert seen == ["America/Chicago"]
    migrated = [entry for entry in logs if entry["event"] == "store.job_schedule_migrated"]
    assert [entry["job"] for entry in migrated] == ["jabcd1234"]
    events = [entry["event"] for entry in logs]
    assert events.index("store.job_schedule_migrated") < events.index("gateway.signal_started")


def _clock_gateway(
    *,
    kernel: FakeClockSync,
    channel: FakeChannel | None = None,
    tmp_path: Path | None = None,
    now: FakeClock | None = None,
    wait_seconds: float = 120.0,
) -> Gateway:
    """A Signal gateway with a fake kernel clock. With ``tmp_path`` it opens a real store."""
    overrides: dict[str, object] = {}
    if tmp_path is not None:
        overrides = {"state_dir": tmp_path, "agents_config_path": _agents_toml(tmp_path)}
    return Gateway(
        signal_settings(
            heartbeat_seconds=0.01,
            FIREWORKS_API_KEY=None,
            timezone="America/Chicago",
            **overrides,
        ),
        signal=FakeSignal(),
        channel=channel,
        clock_sync=kernel,
        now=now or FakeClock(1_800_000_000.0),
        clock_wait_seconds=wait_seconds,
        clock_poll_seconds=0.01,
    )


def _remember(state_dir: Path, *, at: float) -> None:
    """Leave a message in the store, as a gateway that ran at ``at`` would have."""
    store = Store(store_path(state_dir))
    store.save_history("jacob", [Message(role="user", content="hi", created_at=at)])
    store.close()


async def _run_briefly(gateway: Gateway, *, seconds: float = 0.05) -> None:
    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(seconds)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)
