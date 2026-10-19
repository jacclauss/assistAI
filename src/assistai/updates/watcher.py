"""Diff image pins on a schedule and text the operator. Never installs anything."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Protocol

import structlog

from assistai.config import Settings
from assistai.errors import AssistAIError, ManifestError, StoreError, UpdateError
from assistai.manifest import ImagePin, load_manifest, resolve_manifest_path
from assistai.store import Store
from assistai.updates.check import TagPage, UpdateFinding, assess, format_report
from assistai.updates.hub import DockerHubTags

log = structlog.get_logger(__name__)


class TagSource(Protocol):
    async def tags(self, repository: str) -> TagPage: ...

    async def aclose(self) -> None: ...


class UpdateSender(Protocol):
    async def send(self, recipient: str, text: str) -> None: ...


class UpdateWatcher:
    """One loop beside the job runner. A finding is a message, not a pull."""

    def __init__(
        self,
        settings: Settings,
        store: Store,
        signal: UpdateSender,
        *,
        source: TagSource | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._signal = signal
        self._source = source or DockerHubTags()
        self._owns_source = source is None
        self._clock = clock or time.time
        self._interval = settings.update_check_seconds

    async def run(self, stop: asyncio.Event) -> None:
        try:
            while not stop.is_set():
                try:
                    await self._check_unless_stopped(stop)
                except Exception:
                    log.exception("updates.check_failed")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self._interval)
                except TimeoutError:
                    continue
        finally:
            if self._owns_source:
                await self._source.aclose()

    async def check(self) -> str:
        """Return the summary that was sent, or empty when there was nothing new to send."""
        try:
            pins = load_manifest(resolve_manifest_path(self._settings)).images
        except ManifestError:
            log.exception("updates.manifest_unreadable")
            return ""
        findings: list[UpdateFinding] = []
        failed = False
        for pin in pins:
            try:
                found = await self._one(pin)
            except UpdateError:
                log.warning("updates.upstream_failed", image=pin.name)
                failed = True
                continue
            if found is not None:
                findings.append(found)
        report = format_report(findings)
        try:
            previous = self._store.last_update_report()
        except StoreError:
            log.exception("updates.notice_unreadable")
            return ""
        if failed:
            return await self._report_confirmed(findings, previous)
        if report == previous:
            return ""
        if report:
            operator = self._operator()
            if operator is None:
                log.warning("updates.no_operator")
                return ""
            try:
                await self._signal.send(operator, report)
            except AssistAIError:
                log.exception("updates.send_failed")
                return ""
        try:
            self._store.save_update_report(report, self._clock())
        except StoreError:
            log.exception("updates.notice_unsaved")
        return report

    async def _report_confirmed(self, findings: list[UpdateFinding], previous: str) -> str:
        """A failed lookup must not erase a notice or pretend the check finished."""
        fresh = [
            item for item in findings if item.priority == "high" and item.detail not in previous
        ]
        report = format_report(fresh)
        if not report:
            return ""
        operator = self._operator()
        if operator is None:
            log.warning("updates.no_operator")
            return ""
        try:
            await self._signal.send(operator, report)
        except AssistAIError:
            log.exception("updates.send_failed")
            return ""
        combined = f"{previous}\n{report}" if previous else report
        try:
            self._store.save_update_report(combined, self._clock())
        except StoreError:
            log.exception("updates.notice_unsaved")
        return report

    async def _check_unless_stopped(self, stop: asyncio.Event) -> None:
        task = asyncio.create_task(self.check())
        stopped = asyncio.create_task(stop.wait())
        try:
            done, _pending = await asyncio.wait(
                {task, stopped}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            task.cancel()
            stopped.cancel()
            await asyncio.gather(task, stopped, return_exceptions=True)
            raise
        if stopped in done:
            task.cancel()
            await asyncio.gather(task, stopped, return_exceptions=True)
            return
        stopped.cancel()
        await asyncio.gather(stopped, return_exceptions=True)
        await task

    async def _one(self, pin: ImagePin) -> UpdateFinding | None:
        page = await self._source.tags(pin.repository)
        found = assess(pin, page.tags, complete=page.complete)
        if found is None and not pin.digest:
            log.info("updates.unverified", image=pin.name)
        return found

    def _operator(self) -> str | None:
        """The first allowlisted number. Paired guests are not operators."""
        allow = self._settings.allow_from
        if not allow:
            return None
        return allow[0]
