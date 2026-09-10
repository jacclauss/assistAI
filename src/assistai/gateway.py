"""The gateway daemon.

Owns process lifecycle, Fireworks validation, and the Signal receive loop.
Agent routing and the tool broker attach here.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from typing import Protocol

import structlog

from assistai import __version__
from assistai.agents import Household, load_household, resolve_household_path
from assistai.broker import ToolBroker, builtin_catalog
from assistai.config import Settings
from assistai.inference.client import FireworksClient
from assistai.manifest import Manifest, load_manifest, resolve_manifest_path
from assistai.signal.channel import SignalChannel
from assistai.signal.client import SignalClient, SignalTransport
from assistai.signal.policy import AccessPolicy
from assistai.store import Store, store_path


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
        self._store: Store | None = None
        self._owns_store = False
        self._channel_task: asyncio.Task[None] | None = None
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
        if self._channel is None:
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
        self._channel_task = asyncio.create_task(self._run_channel(self._channel))
        log.info("gateway.signal_started", account=self._settings.signal_account)

    async def _run_channel(self, channel: ChannelRunner) -> None:
        try:
            await channel.run(self._shutdown)
        except Exception:
            log.exception("signal.channel_failed")
            self.request_shutdown("signal_channel_failed")

    async def _stop_signal(self) -> None:
        if self._channel_task is None:
            return
        self._shutdown.set()
        try:
            await asyncio.wait_for(
                self._channel_task,
                timeout=self._settings.shutdown_grace_seconds,
            )
        except TimeoutError:
            self._channel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._channel_task
        self._channel_task = None

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
