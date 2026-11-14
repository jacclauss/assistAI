"""The gateway daemon.

Owns process lifecycle, Fireworks validation, and the Signal receive loop.
Agent routing and the tool broker attach here.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from collections.abc import Callable
from typing import Protocol
from zoneinfo import ZoneInfo

import structlog

from assistai import __version__
from assistai.agents import Household, load_household, resolve_household_path
from assistai.broker import ToolBroker, builtin_catalog
from assistai.calendar.tools import bind_calendar
from assistai.clock import HouseholdClock, local_reading
from assistai.config import TIMEZONE_REQUIRED, Settings
from assistai.errors import HouseholdConfigError
from assistai.inference.client import FireworksClient
from assistai.mail.tools import bind_mail
from assistai.manifest import Manifest, load_manifest, resolve_manifest_path
from assistai.research.tools import assert_isolated_extract, bind_research
from assistai.scheduler import JobRunner
from assistai.secrets import apply_caldav_password
from assistai.shared.organizer import Organizer
from assistai.shared.tools import bind_shared
from assistai.signal.channel import SignalChannel
from assistai.signal.client import SignalClient, SignalTransport
from assistai.signal.policy import AccessPolicy
from assistai.store import Store, store_path
from assistai.sysclock import ClockState, ClockSync, KernelClockSync
from assistai.updates.watcher import UpdateWatcher

# How long Signal waits for a believable clock. After that an unsynced kernel
# alone starts anyway, since a household with no bot is worse. A clock still
# behind the store without a synced kernel refuses instead: history saved on
# it is deleted once NTP steps it forward.
CLOCK_WAIT_SECONDS = 120.0
CLOCK_BEHIND_REFUSED = (
    "the clock is behind the newest stored event and the kernel does not say it "
    "is synchronized, so Signal is not starting; the container restart waits again"
)
# Docker kills ten seconds after SIGTERM, so the wait checks for shutdown
# far more often than that.
CLOCK_POLL_SECONDS = 1.0
# NTP corrections and a slow shutdown are minutes at most. A clock further
# behind the newest stored event than this has not been set yet.
CLOCK_BEHIND_SECONDS = 300.0
# A stored time this far ahead is a bad row, not a slow clock. Believing it
# would hold the gateway down on every start.
STORE_FUTURE_SECONDS = 24 * 3600.0


class ChannelRunner(Protocol):
    async def run(self, stop: asyncio.Event) -> None: ...


log = structlog.get_logger(__name__)


class Gateway:
    """Long-lived process that owns channel connections, routing, and policy."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: FireworksClient | None = None,
        manifest: Manifest | None = None,
        signal: SignalTransport | None = None,
        channel: ChannelRunner | None = None,
        household: Household | None = None,
        broker: ToolBroker | None = None,
        clock_sync: ClockSync | None = None,
        now: Callable[[], float] = time.time,
        clock_wait_seconds: float = CLOCK_WAIT_SECONDS,
        clock_poll_seconds: float = CLOCK_POLL_SECONDS,
    ) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None
        self._manifest = manifest
        self._signal = signal
        self._owns_signal = signal is None
        self._channel = channel
        self._household = household
        self._broker = broker
        self._clock_sync: ClockSync = clock_sync or KernelClockSync()
        self._now = now
        self._clock_wait_seconds = clock_wait_seconds
        self._clock_poll_seconds = clock_poll_seconds
        self._clock: HouseholdClock | None = None
        self._store: Store | None = None
        self._owns_store = False
        self._channel_task: asyncio.Task[None] | None = None
        self._job_task: asyncio.Task[None] | None = None
        self._jobs: JobRunner | None = None
        self._update_task: asyncio.Task[None] | None = None
        self._updates: UpdateWatcher | None = None
        self._shutdown = asyncio.Event()
        self._beats = 0
        self.validated_refs: tuple[str, ...] = ()

    @property
    def beats(self) -> int:
        """Heartbeats emitted so far. Exposed for tests and diagnostics."""
        return self._beats

    def request_shutdown(self, reason: str) -> None:
        """Ask the run loop to stop. Safe to call more than once."""
        if not self._shutdown.is_set():
            log.info("shutdown.requested", reason=reason)
            self._shutdown.set()

    async def run(self) -> None:
        """Run until shutdown is requested."""
        log.info(
            "gateway.starting",
            version=__version__,
            heartbeat_seconds=self._settings.heartbeat_seconds,
        )
        try:
            await self._validate_models()
            await self._start_signal()
            await self._heartbeat_loop()
        finally:
            await self._stop_signal()
            await self._close_owned()
            log.info("gateway.stopped", beats=self._beats)

    async def _validate_models(self) -> None:
        """Confirm pinned models are callable. Skipped when no API key is set."""
        if self._settings.fireworks_api_key is None:
            log.warning("gateway.skip_model_validation", reason="no_api_key")
            return
        if self._client is None:
            self._client = FireworksClient(self._settings)
            self._owns_client = True
        manifest = self._manifest or load_manifest(resolve_manifest_path(self._settings))
        self._manifest = manifest
        refs = manifest.required_refs()
        await self._client.validate(refs)
        self.validated_refs = refs
        log.info("gateway.models_validated", refs=list(refs))

    async def _start_signal(self) -> None:
        if self._settings.signal_account is None:
            log.info("gateway.skip_signal", reason="no_account")
            return
        self._clock = self._household_clock()
        if self._channel is None:
            # Keychain can wait on an Allow prompt. Run it off the event loop
            # so a shutdown signal is still delivered, and fail before Signal
            # or the store are opened.
            self._settings = await asyncio.to_thread(apply_caldav_password, self._settings)
            if self._signal is None:
                self._signal = SignalClient(self._settings)
                self._owns_signal = True
            await self._signal.wait_until_healthy(
                self._settings.signal_startup_timeout_seconds, self._shutdown
            )
            await self._signal.check()
            if not self._settings.allow_from:
                log.warning(
                    "gateway.signal_allowlist_empty",
                    reason="set ASSISTAI_SIGNAL_ALLOW_FROM or nobody can text the bot",
                )
            household = self._household or load_household(
                resolve_household_path(self._settings),
                known_tools=builtin_catalog().names(),
            )
            self._household = household
            assert_isolated_extract(
                self._settings,
                {tool for agent in household.agents for tool in agent.tools},
            )
            unbound = [
                number
                for number in self._settings.allow_from
                if household.agent_for_signal_dm(number) is None
            ]
            if unbound:
                log.warning("gateway.allow_from_unbound", numbers=unbound)
            broker = self._broker or ToolBroker(builtin_catalog(), household.broker)
            self._broker = broker
            store = Store(store_path(self._settings.state_dir))
            self._store = store
            self._owns_store = True
            store.import_legacy_allowlist(self._settings.state_dir / "signal-allowlist.json")
            policy = AccessPolicy(
                self._settings.allow_from,
                store=store,
                pairing_ttl_seconds=self._settings.signal_pairing_ttl_seconds,
            )
            self._channel = SignalChannel(
                self._settings,
                self._signal,
                policy=policy,
                fireworks=self._client,
                manifest=self._manifest,
                household=household,
                broker=broker,
                store=store,
            )
            for name, handler in bind_research(
                self._settings,
                fireworks=self._client,
                quarantine=self._manifest.quarantine if self._manifest is not None else None,
            ).items():
                broker.bind(name, handler)
            for name, handler in bind_calendar(self._settings, clock=self._clock).items():
                broker.bind(name, handler)
            for name, handler in bind_mail(self._settings, store).items():
                broker.bind(name, handler)
            for name, handler in bind_shared(Organizer(store)).items():
                broker.bind(name, handler)
            self._jobs = JobRunner(
                store, household, self._channel, self._settings, clock=self._clock.now
            )
            if self._signal is not None:
                self._updates = UpdateWatcher(self._settings, store, self._signal)
        if not await self._wait_for_clock():
            return
        if self._store is not None and self._clock is not None:
            # Every start, before the runner reads a row: a row written by an
            # older build, before an upgrade or during a rollback, has no
            # schedule yet. After the clock wait, because a stepped row's next
            # run is chosen from now.
            self._store.backfill_job_schedules(self._clock.zone.key, now=self._clock.now())
        self._channel_task = asyncio.create_task(self._run_channel(self._channel))
        if self._jobs is not None:
            self._job_task = asyncio.create_task(self._jobs.run(self._shutdown))
        if self._updates is not None:
            self._update_task = asyncio.create_task(self._updates.run(self._shutdown))
        log.info("gateway.signal_started", account=self._settings.signal_account)

    def _household_clock(self) -> HouseholdClock:
        """The household zone, or a refusal to start Signal without one."""
        name = self._settings.timezone
        if name is None:
            log.error("gateway.timezone_missing", reason=TIMEZONE_REQUIRED)
            raise HouseholdConfigError(TIMEZONE_REQUIRED)
        clock = HouseholdClock(zone=ZoneInfo(name), now=self._now)
        local, offset = local_reading(clock.now(), clock=clock)
        log.info("clock.zone", zone=name, local=local, utc_offset=offset)
        return clock

    async def _wait_for_clock(self) -> bool:
        """Hold Signal until the clock is believable, at most the configured wait.

        Times written on a clock that is behind stay wrong, and when NTP steps
        it forward, fresh history looks old enough to delete. The clock is
        believed when the kernel says it is synchronized (or cannot say) and
        it is not more than five minutes behind the newest event in the store.
        A row more than a day ahead is ignored only while the kernel says
        synchronized, so a bad row cannot hold startup and a clock days behind
        cannot pass as an empty store. When the wait runs out with the kernel
        not synchronized and the clock still behind the store, it raises
        HouseholdConfigError so the process exits and the restart waits again;
        any other doubtful clock starts anyway, since a household with no bot is
        worse. Returns False when shutdown arrives first.
        """
        newest = self._store.newest_event_at() if self._store is not None else None
        loop = asyncio.get_running_loop()
        # The wait is timed on the monotonic clock: the wall clock is the
        # thing that may step while we wait.
        started = loop.time()
        deadline = started + self._clock_wait_seconds
        waiting = False
        unknown_logged = False
        future_logged = False
        while True:
            state = self._clock_sync.state()
            if state is ClockState.UNKNOWN and not unknown_logged:
                log.warning("clock.sync_unknown", reason="the kernel did not report clock state")
                unknown_logged = True
            behind = newest - self._now() if newest is not None else 0.0
            # Only a synced kernel can call a far-ahead row bad. Unsynced or
            # unknown, the same row may mean the clock is days behind, and
            # dropping it would let that clock through.
            if state is ClockState.SYNCED and behind > STORE_FUTURE_SECONDS:
                if not future_logged:
                    log.warning("clock.store_in_future", ahead_seconds=round(behind))
                    future_logged = True
                behind = 0.0
            kernel_ok = state is not ClockState.UNSYNCED
            store_ok = behind <= CLOCK_BEHIND_SECONDS
            if kernel_ok and store_ok:
                log.info(
                    "clock.synced",
                    kernel=state.value,
                    waited_seconds=round(loop.time() - started, 1),
                )
                return True
            if not waiting:
                log.info(
                    "clock.waiting_for_sync",
                    kernel=state.value,
                    behind_store_seconds=round(max(behind, 0.0)),
                    wait_seconds=self._clock_wait_seconds,
                )
                waiting = True
            remaining = deadline - loop.time()
            if remaining <= 0:
                if state is not ClockState.SYNCED and not store_ok:
                    log.error(
                        "clock.unsynced",
                        kernel=state.value,
                        behind_store_seconds=round(behind),
                        reason=CLOCK_BEHIND_REFUSED,
                    )
                    raise HouseholdConfigError(CLOCK_BEHIND_REFUSED)
                log.warning(
                    "clock.unsynced",
                    kernel=state.value,
                    behind_store_seconds=round(max(behind, 0.0)),
                    reason="starting anyway on a clock that may be wrong",
                )
                return True
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=min(self._clock_poll_seconds, remaining)
                )
            if self._shutdown.is_set():
                return False

    async def _run_channel(self, channel: ChannelRunner) -> None:
        try:
            await channel.run(self._shutdown)
        except Exception:
            log.exception("signal.channel_failed")
            self.request_shutdown("signal_channel_failed")

    async def _stop_signal(self) -> None:
        self._shutdown.set()
        await self._await_task(self._channel_task)
        await self._await_task(self._job_task)
        await self._await_task(self._update_task)
        self._channel_task = None
        self._job_task = None
        self._update_task = None

    async def _await_task(self, task: asyncio.Task[None] | None) -> None:
        if task is None:
            return
        try:
            await asyncio.wait_for(
                task,
                timeout=self._settings.shutdown_grace_seconds,
            )
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _close_owned(self) -> None:
        if self._owns_signal and self._signal is not None:
            await self._signal.aclose()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        if self._owns_store and self._store is not None:
            self._store.close()

    async def _heartbeat_loop(self) -> None:
        interval = self._settings.heartbeat_seconds
        while not self._shutdown.is_set():
            self._beats += 1
            log.info("gateway.heartbeat", beat=self._beats)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._shutdown.wait(), timeout=interval)


def install_signal_handlers(gateway: Gateway) -> None:
    """Route SIGINT and SIGTERM into a graceful shutdown.

    Container runtimes send SIGTERM and then kill after a grace period, so the
    daemon must not treat termination as an error path.
    """
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, gateway.request_shutdown, sig.name)
