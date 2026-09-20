"""In-process job runner. Few jobs, one box; a second daemon buys nothing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Literal, Protocol

import structlog

from assistai.agents import AgentSpec, Household
from assistai.config import Settings
from assistai.errors import StoreError
from assistai.jobs import (
    Job,
    JobOutcome,
    format_expire_message,
    format_fail_message,
    format_job_message,
)
from assistai.store import Store

log = structlog.get_logger(__name__)

NotifyStatus = Literal["sent", "rate_limited", "failed"]
_AfterNotify = Literal["advance", "cancel"]

# Signal answers a hammered send with 429, and the poll is every second, so a
# delivery that fails waits before it tries again.
_FIRST_RETRY_SECONDS = 30.0
_MAX_RETRY_SECONDS = 600.0


class JobChannel(Protocol):
    def can_notify_job(self, agent: AgentSpec) -> bool: ...

    async def run_job(self, agent: AgentSpec, job: Job) -> JobOutcome: ...

    async def notify_job(self, agent: AgentSpec, text: str, *, untrusted: bool) -> NotifyStatus: ...

    async def persist_job_report(self, agent: AgentSpec, text: str, *, untrusted: bool) -> bool: ...


class JobRunner:
    """Poll SQLite for due jobs and report to the owner over Signal."""

    def __init__(
        self,
        store: Store,
        household: Household,
        channel: JobChannel,
        settings: Settings,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._household = household
        self._channel = channel
        self._poll = settings.job_poll_seconds
        self._clock = clock or time.time
        self._running: set[str] = set()
        self._unknown_warned: set[str] = set()
        self._retry_at: dict[str, float] = {}
        self._retry_delay: dict[str, float] = {}

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self._tick()
            except StoreError:
                log.exception("jobs.poll_failed")
            except Exception:
                log.exception("jobs.tick_failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._poll)
            except TimeoutError:
                continue

    async def _tick(self) -> None:
        now = self._clock()
        due = self._store.due_jobs(now)
        for job in due:
            if job.id in self._running or now < self._retry_at.get(job.id, 0.0):
                continue
            agent = self._agent(job.agent)
            if agent is None:
                if job.id not in self._unknown_warned:
                    log.warning("jobs.unknown_agent", job=job.id, agent=job.agent)
                    self._unknown_warned.add(job.id)
                continue
            self._unknown_warned.discard(job.id)
            self._running.add(job.id)
            try:
                await self._fire(job, agent, now)
            except Exception:
                # One unwritable row must not hold up everyone else's schedule,
                # or call the model again on the next one-second poll.
                log.exception("jobs.fire_failed", job=job.id, name=job.name)
                self._defer(job.id)
            finally:
                self._running.discard(job.id)

    async def _fire(self, job: Job, agent: AgentSpec, now: float) -> None:
        if job.pending_text is not None and job.pending_after is not None:
            await self._deliver(job, agent)
            return

        expired = job.kind == "watch" and job.expires_at is not None and job.expires_at <= now
        due = job.next_run_at <= now
        # Only the runs that always talk are worth paying a model call for when
        # the send budget is gone. A quiet watch still needs to look, or one
        # chatty schedule would stop every watch on the box.
        if (job.kind == "schedule" or expired) and not self._channel.can_notify_job(agent):
            log.warning("jobs.rate_limited", job=job.id, name=job.name)
            return
        if expired and not due:
            await self._queue(
                job,
                agent,
                format_expire_message(job.name),
                untrusted=False,
                after="cancel",
            )
            return

        after: _AfterNotify = "cancel" if expired else "advance"
        try:
            outcome = await self._channel.run_job(agent, job)
        except Exception:
            # Auth, network, tool error, or a bug in the loop: the owner is told
            # either way. No silent skip. A watch whose TTL is already up must
            # not stay on the list after that ping.
            log.exception("jobs.run_failed", job=job.id, name=job.name)
            await self._queue(
                job,
                agent,
                format_fail_message(job.name),
                untrusted=False,
                after=after,
            )
            return

        if job.kind == "schedule" or outcome.found:
            body = outcome.text.strip() if outcome.found else "Nothing to report."
            await self._queue(
                job,
                agent,
                format_job_message(job.name, body),
                untrusted=outcome.untrusted,
                after=after,
            )
            return
        if expired:
            await self._queue(
                job,
                agent,
                format_expire_message(job.name),
                untrusted=False,
                after="cancel",
            )
            return
        self._store.advance_job(job.id, now=self._clock())
        log.info("jobs.ran", job=job.id, name=job.name, kind=job.kind, found=False)

    async def _queue(
        self,
        job: Job,
        agent: AgentSpec,
        text: str,
        *,
        untrusted: bool,
        after: _AfterNotify,
    ) -> None:
        queued = replace(
            job,
            pending_text=text,
            pending_untrusted=untrusted,
            pending_after=after,
            pending_delivered=False,
        )
        if not self._store.queue_job_report(queued):
            log.info("jobs.cancelled_before_send", job=job.id, name=job.name)
            return
        await self._deliver(queued, agent)

    async def _deliver(self, job: Job, agent: AgentSpec) -> None:
        text = job.pending_text
        after = job.pending_after
        if text is None or after is None:
            return
        if not job.pending_delivered:
            status = await self._channel.notify_job(agent, text, untrusted=job.pending_untrusted)
            if status != "sent":
                if status == "rate_limited":
                    log.warning("jobs.rate_limited", job=job.id, name=job.name)
                else:
                    log.warning("jobs.notify_failed", job=job.id, name=job.name)
                self._defer(job.id)
                return
            if not self._store.mark_job_delivered(job.id):
                # Outbox vanished while Signal was sending. Persist once so
                # history matches the text that went out; there is nothing left
                # to retry.
                log.info("jobs.cancelled_mid_send", job=job.id, name=job.name)
                self._clear_backoff(job.id)
                await self._channel.persist_job_report(agent, text, untrusted=job.pending_untrusted)
                return
        if not await self._channel.persist_job_report(agent, text, untrusted=job.pending_untrusted):
            log.warning("jobs.history_unpersisted", job=job.id, name=job.name)
            self._defer(job.id)
            return
        self._clear_backoff(job.id)
        finished = self._clock()
        if after == "cancel":
            self._store.cancel_job(job.id, at=finished)
            self._store.clear_job_outbox(job.id)
            log.info("jobs.expired", job=job.id, name=job.name)
            return
        if not self._store.advance_job(job.id, now=finished):
            self._store.clear_job_outbox(job.id)
            return
        log.info("jobs.ran", job=job.id, name=job.name, kind=job.kind)

    def _defer(self, job_id: str) -> None:
        """Hold off the next attempt, doubling each time it fails."""
        delay = self._retry_delay.get(job_id, _FIRST_RETRY_SECONDS)
        self._retry_at[job_id] = self._clock() + delay
        self._retry_delay[job_id] = min(delay * 2, _MAX_RETRY_SECONDS)

    def _clear_backoff(self, job_id: str) -> None:
        self._retry_at.pop(job_id, None)
        self._retry_delay.pop(job_id, None)

    def _agent(self, name: str) -> AgentSpec | None:
        for agent in self._household.agents:
            if agent.name == name:
                return agent
        return None
