"""The gateway daemon.

Phase 0 implements the process lifecycle only: start, heartbeat, and graceful
shutdown. Channel connections, agent routing, and the tool broker attach here in
later phases.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

import structlog

from assistai import __version__
from assistai.config import Settings
from assistai.inference.client import FireworksClient
from assistai.manifest import Manifest, load_manifest, resolve_manifest_path

log = structlog.get_logger(__name__)


class Gateway:
    """Long-lived process that will own channel connections, routing, and policy."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: FireworksClient | None = None,
        manifest: Manifest | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._manifest = manifest
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
            await self._heartbeat_loop()
        finally:
            log.info("gateway.stopped", beats=self._beats)

    async def _validate_models(self) -> None:
        """Confirm pinned models are callable. Skipped when no API key is set."""
        if self._settings.fireworks_api_key is None:
            log.warning("gateway.skip_model_validation", reason="no_api_key")
            return
        owns_client = self._client is None
        client = self._client or FireworksClient(self._settings)
        manifest = self._manifest or load_manifest(resolve_manifest_path(self._settings))
        refs = manifest.required_refs()
        try:
            await client.validate(refs)
        finally:
            if owns_client:
                await client.aclose()
        self.validated_refs = refs
        log.info("gateway.models_validated", refs=list(refs))

    async def _heartbeat_loop(self) -> None:
        interval = self._settings.heartbeat_seconds
        while not self._shutdown.is_set():
            self._beats += 1
            log.info("gateway.heartbeat", beat=self._beats)
            # Returns early when shutdown fires, so SIGTERM does not wait out
            # the full interval.
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
