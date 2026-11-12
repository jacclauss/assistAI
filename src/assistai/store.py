"""Durable household state. One SQLite file in the gateway state volume.

History, untrusted labels, the pairing allowlist, staged proposals, jobs,
and per-person Gmail refresh tokens live here so a reboot cannot forget a
conversation, silently clear taint, skip a confirmation, or drop a schedule.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import structlog

from assistai.clock import (
    HouseholdClock,
    Schedule,
    local_reading,
    next_slot,
    shifted_by_spring_gap,
    slots_between,
)
from assistai.errors import JobError, StoreError
from assistai.inference.types import Message, ToolCall
from assistai.jobs import DAY_SECONDS, Job, JobKind, schedule_from_seconds
from assistai.signal.numbers import InvalidNumberError, normalize_e164
from assistai.staging import Proposal

log = structlog.get_logger(__name__)

SCHEMA_VERSION = 15
_DB_NAME = "assistai.sqlite"
# A proposal stamped more than this far ahead of now was made before the
# clock stepped back. Its ten minutes can no longer be measured, so it is
# expired rather than left confirmable for as long as the step was.
_BACKWARD_STEP_SECONDS = 60.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    untrusted INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE(agent, seq)
);
CREATE INDEX IF NOT EXISTS messages_agent_seq ON messages(agent, seq);

CREATE TABLE IF NOT EXISTS allowlist (
    number TEXT PRIMARY KEY NOT NULL,
    admitted_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    agent TEXT PRIMARY KEY NOT NULL,
    calls TEXT NOT NULL,
    tainted INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY NOT NULL,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    every_seconds INTEGER NOT NULL,
    ttl_seconds INTEGER,
    created_at REAL NOT NULL,
    expires_at REAL,
    next_run_at REAL NOT NULL,
    last_run_at REAL,
    cancelled_at REAL,
    pending_text TEXT,
    pending_untrusted INTEGER NOT NULL DEFAULT 0,
    pending_after TEXT,
    pending_delivered INTEGER NOT NULL DEFAULT 0,
    once INTEGER NOT NULL DEFAULT 0,
    verbatim INTEGER NOT NULL DEFAULT 0,
    relay INTEGER NOT NULL DEFAULT 0,
    relay_sent INTEGER NOT NULL DEFAULT 0,
    relay_recorded INTEGER NOT NULL DEFAULT 0,
    tainted INTEGER NOT NULL DEFAULT 0,
    zone TEXT,
    anchor_at REAL,
    every_days INTEGER,
    at_local TEXT
);
CREATE INDEX IF NOT EXISTS jobs_due ON jobs(cancelled_at, next_run_at);
CREATE INDEX IF NOT EXISTS jobs_agent_name ON jobs(agent, name);

CREATE TABLE IF NOT EXISTS gmail_tokens (
    agent TEXT PRIMARY KEY NOT NULL,
    refresh_token TEXT NOT NULL,
    access_token TEXT,
    expires_at REAL NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_agent_active_name
    ON jobs(agent, lower(name)) WHERE cancelled_at IS NULL;

CREATE TABLE IF NOT EXISTS shared_lists (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    owner TEXT,
    created_by TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS shared_lists_shared_name
    ON shared_lists(lower(name)) WHERE owner IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS shared_lists_private_name
    ON shared_lists(owner, lower(name)) WHERE owner IS NOT NULL;

CREATE TABLE IF NOT EXISTS shared_items (
    id TEXT PRIMARY KEY NOT NULL,
    list_id TEXT NOT NULL REFERENCES shared_lists(id),
    text TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    created_by TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS update_notices (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    report TEXT NOT NULL,
    sent_at REAL NOT NULL
);
"""

_PERSISTED_ROLES = frozenset({"user", "assistant", "tool"})
# Added in version 15. Every one is nullable and no version 14 query names
# it, which is what makes ``downgrade(path, to=14)`` safe.
_SCHEDULE_COLUMNS = (
    ("zone", "TEXT"),
    ("anchor_at", "REAL"),
    ("every_days", "INTEGER"),
    ("at_local", "TEXT"),
)


def store_path(state_dir: Path) -> Path:
    return state_dir / _DB_NAME


class Store:
    """Synchronous SQLite access. The gateway is single-threaded asyncio."""

    def __init__(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, timeout=5.0)
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"state database unreadable: {path}") from exc
        self._path = path
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
        except sqlite3.Error as exc:
            self._conn.close()
            raise StoreError(f"state database unreadable: {path}") from exc
        except StoreError:
            self._conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def load_history(self, agent: str) -> list[Message]:
        """Non-system messages for ``agent``, oldest first.

        A corrupt row fails closed: returning an empty list would launder taint.
        """
        try:
            rows = self._conn.execute(
                "SELECT role, content, tool_calls, tool_call_id, untrusted, created_at "
                "FROM messages WHERE agent = ? ORDER BY seq ASC",
                (agent,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"history for {agent} is unreadable") from exc
        messages: list[Message] = []
        for row in rows:
            messages.append(_message_from_row(row, agent=agent))
        return messages

    def save_history(self, agent: str, messages: list[Message]) -> None:
        """Replace persisted history. System messages are not stored."""
        self.persist(agent, messages)

    def persist(
        self,
        agent: str,
        messages: list[Message],
        *,
        proposal: Proposal | Literal["keep"] | None = "keep",
    ) -> None:
        """Write history and optionally replace or clear the live proposal."""
        persisted = [message for message in messages if message.role in _PERSISTED_ROLES]
        try:
            with self._conn:
                self._replace_history(agent, persisted)
                if proposal != "keep":
                    self._replace_proposal_row(agent, proposal)
        except sqlite3.Error as exc:
            raise StoreError(f"history for {agent} could not be saved") from exc

    def load_proposal(self, agent: str, *, now: float | None = None) -> Proposal | None:
        """The live proposal, or None if missing or expired."""
        proposal, _expired = self.take_proposal(agent, now=now)
        return proposal

    def take_proposal(
        self, agent: str, *, now: float | None = None
    ) -> tuple[Proposal | None, bool]:
        """Return the live proposal and whether an expired row was dropped.

        An expired row is deleted so a later yes cannot fire it. A row made
        more than a minute in the future counts as expired: the clock stepped
        back since, and a stale yes must not outlive its ten minutes by the
        size of the step. Corrupt JSON fails closed: executing garbage would
        be worse than asking again.
        """
        try:
            row = self._conn.execute(
                "SELECT calls, tainted, created_at, expires_at FROM proposals WHERE agent = ?",
                (agent,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"proposal for {agent} is unreadable") from exc
        if row is None:
            return None, False
        expires_at = row["expires_at"]
        created_at = row["created_at"]
        if not isinstance(expires_at, (int, float)) or not isinstance(created_at, (int, float)):
            raise StoreError(f"proposal for {agent} is unreadable")
        moment = now if now is not None else time.time()
        if float(expires_at) <= moment or moment < float(created_at) - _BACKWARD_STEP_SECONDS:
            self.clear_proposal(agent)
            return None, True
        tainted = row["tainted"]
        if tainted not in (0, 1):
            raise StoreError(f"proposal for {agent} is unreadable")
        calls = _proposal_calls(row["calls"], agent=agent)
        return (
            Proposal(
                agent=agent,
                calls=tuple(calls),
                tainted=bool(tainted),
                created_at=float(created_at),
                expires_at=float(expires_at),
            ),
            False,
        )

    def newest_event_at(self) -> float | None:
        """The latest moment the store saw something happen, or None when it has nothing.

        Only past events count: messages, proposals, job creations, and job
        runs. ``next_run_at`` and ``expires_at`` are in the future by design
        and would hold startup for no reason.
        """
        try:
            row = self._conn.execute(
                "SELECT MAX(moment) AS newest FROM ("
                "SELECT MAX(created_at) AS moment FROM messages "
                "UNION ALL SELECT MAX(created_at) FROM proposals "
                "UNION ALL SELECT MAX(created_at) FROM jobs "
                "UNION ALL SELECT MAX(last_run_at) FROM jobs)"
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError("state database is unreadable") from exc
        newest = row["newest"] if row is not None else None
        if newest is None:
            return None
        if not isinstance(newest, (int, float)):
            raise StoreError("state database is unreadable")
        return float(newest)

    def replace_proposal(self, proposal: Proposal) -> None:
        try:
            with self._conn:
                self._replace_proposal_row(proposal.agent, proposal)
        except sqlite3.Error as exc:
            raise StoreError(f"proposal for {proposal.agent} could not be saved") from exc

    def clear_proposal(self, agent: str) -> None:
        try:
            with self._conn:
                self._conn.execute("DELETE FROM proposals WHERE agent = ?", (agent,))
        except sqlite3.Error as exc:
            raise StoreError(f"proposal for {agent} could not be saved") from exc

    def approved(self) -> frozenset[str]:
        try:
            rows = self._conn.execute("SELECT number FROM allowlist").fetchall()
        except sqlite3.Error as exc:
            raise StoreError("allowlist is unreadable") from exc
        return frozenset(row["number"] for row in rows)

    def admit(self, number: str) -> None:
        normalized = normalize_e164(number)
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT OR IGNORE INTO allowlist (number, admitted_at) VALUES (?, ?)",
                    (normalized, time.time()),
                )
        except sqlite3.Error as exc:
            raise StoreError("allowlist could not be updated") from exc

    def import_legacy_allowlist(self, path: Path) -> None:
        """Copy pairing approvals from the phase-2 JSON file, then retire it."""
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("store.legacy_allowlist_unreadable", error=type(exc).__name__)
            return
        rows = raw.get("approved") if isinstance(raw, dict) else None
        if not isinstance(rows, list):
            log.warning("store.legacy_allowlist_unreadable", error="shape")
            return
        imported = 0
        for item in rows:
            if not isinstance(item, str):
                continue
            try:
                self.admit(item)
            except InvalidNumberError:
                continue
            except StoreError:
                log.exception("store.legacy_allowlist_incomplete")
                return
            imported += 1
        retired = path.with_name(path.name + ".migrated")
        try:
            path.replace(retired)
        except OSError as exc:
            log.warning("store.legacy_allowlist_not_retired", error=type(exc).__name__)
            return
        log.info("store.legacy_allowlist_imported", count=imported, retired=str(retired))

    def save_job(self, job: Job) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO jobs (id, agent, kind, name, prompt, every_seconds, "
                    "ttl_seconds, created_at, expires_at, next_run_at, last_run_at, "
                    "cancelled_at, pending_text, pending_untrusted, pending_after, "
                    "pending_delivered, once, verbatim, relay, relay_sent, relay_recorded, "
                    "tainted, zone, anchor_at, every_days, at_local) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?)",
                    (
                        job.id,
                        job.agent,
                        job.kind,
                        job.name,
                        job.prompt,
                        job.every_seconds,
                        job.ttl_seconds,
                        job.created_at,
                        job.expires_at,
                        job.next_run_at,
                        job.last_run_at,
                        job.cancelled_at,
                        job.pending_text,
                        1 if job.pending_untrusted else 0,
                        job.pending_after,
                        1 if job.pending_delivered else 0,
                        1 if job.once else 0,
                        1 if job.verbatim else 0,
                        1 if job.relay else 0,
                        1 if job.relay_sent else 0,
                        1 if job.relay_recorded else 0,
                        1 if job.tainted else 0,
                        job.zone,
                        job.anchor_at,
                        job.every_days,
                        job.at_local,
                    ),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"job {job.id} could not be saved") from exc

    def list_jobs(self, agent: str) -> list[Job]:
        try:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE agent = ? AND cancelled_at IS NULL "
                "ORDER BY next_run_at ASC",
                (agent,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError(f"jobs for {agent} are unreadable") from exc
        return _jobs_from_rows(rows)

    def due_jobs(self, now: float) -> list[Job]:
        try:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE "
                "(cancelled_at IS NULL AND ("
                "pending_text IS NOT NULL OR "
                "next_run_at <= ? OR "
                "(kind = 'watch' AND expires_at IS NOT NULL AND expires_at <= ?)"
                ")) OR (pending_text IS NOT NULL AND pending_delivered = 1) "
                "ORDER BY next_run_at ASC",
                (now, now),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError("due jobs are unreadable") from exc
        return _jobs_from_rows(rows)

    def find_job(
        self, agent: str, *, name: str | None = None, job_id: str | None = None
    ) -> Job | None:
        try:
            if job_id is not None and name is not None:
                row = self._conn.execute(
                    "SELECT * FROM jobs WHERE id = ? AND agent = ? AND cancelled_at IS NULL "
                    "AND lower(name) = lower(?)",
                    (job_id, agent, name),
                ).fetchone()
            elif job_id is not None:
                row = self._conn.execute(
                    "SELECT * FROM jobs WHERE id = ? AND agent = ? AND cancelled_at IS NULL",
                    (job_id, agent),
                ).fetchone()
            elif name is not None:
                row = self._conn.execute(
                    "SELECT * FROM jobs WHERE agent = ? AND cancelled_at IS NULL "
                    "AND lower(name) = lower(?)",
                    (agent, name),
                ).fetchone()
            else:
                return None
        except sqlite3.Error as exc:
            raise StoreError(f"jobs for {agent} are unreadable") from exc
        if row is None:
            return None
        return _job_from_row(row)

    def queue_job_report(self, job: Job) -> bool:
        """Store the outbox fields only. False if the job was cancelled meanwhile.

        The runner holds a row it read before a model call, so writing the whole
        row back would revert a cancel or a reschedule the owner asked for while
        the job was running.
        """
        return self._touch_job(
            "UPDATE jobs SET pending_text = ?, pending_untrusted = ?, pending_after = ?, "
            "pending_delivered = 0 WHERE id = ? AND cancelled_at IS NULL",
            (job.pending_text, 1 if job.pending_untrusted else 0, job.pending_after),
            job.id,
        )

    def mark_job_delivered(self, job_id: str) -> bool:
        """Record that the report reached Signal. False if the outbox is already gone.

        Cancel must not block this. The text already went out, and due_jobs only
        retries a cancelled row once pending_delivered is set.
        """
        return self._touch_job(
            "UPDATE jobs SET pending_delivered = 1 WHERE id = ? AND pending_text IS NOT NULL",
            (),
            job_id,
        )

    def advance_job(self, job_id: str, *, now: float) -> bool:
        """Put the next run on the job's next slot and clear the outbox. False if cancelled.

        The next run is the first slot of the stored schedule after ``now``,
        so how long the run took, when it finished, and how often its send
        was retried never move it, and a daily job keeps its hour (decision
        0002). A job that missed slots while the gateway was down runs once
        and then returns to its slot; the slots it skipped are logged, not
        run. The row is read in the same transaction as the write, and a
        stored next run already after ``now`` is a reschedule confirmed
        during the run, which stands. A one-time job has no next slot, so
        advancing one ends it rather than leaving it due on every poll.
        """
        skipped = 0
        shifted: str | None = None
        try:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT * FROM jobs WHERE id = ? AND cancelled_at IS NULL", (job_id,)
                ).fetchone()
                if row is None:
                    return False
                job = _job_from_row(row)
                ended: float | None = None
                next_run_at = job.next_run_at
                if next_run_at <= now:
                    schedule = _stored_schedule(job)
                    if schedule.repeats:
                        next_run_at = next_slot(schedule, after=now)
                        skipped = slots_between(schedule, after=job.next_run_at, upto=now)
                        if shifted_by_spring_gap(schedule, next_run_at):
                            zone = HouseholdClock(zone=schedule.zone)
                            shifted, _offset = local_reading(next_run_at, clock=zone)
                    else:
                        ended = now
                self._conn.execute(
                    "UPDATE jobs SET next_run_at = ?, last_run_at = ?, cancelled_at = ?, "
                    "pending_text = NULL, pending_untrusted = 0, pending_after = NULL, "
                    "pending_delivered = 0, relay_sent = 0, relay_recorded = 0 "
                    "WHERE id = ? AND cancelled_at IS NULL",
                    (next_run_at, now, ended, job_id),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"job {job_id} could not be saved") from exc
        except JobError as exc:
            raise StoreError(f"job {job_id} has no next run") from exc
        # Logged after the commit so a rolled-back advance is not reported.
        if skipped:
            log.info("jobs.slots_skipped", job=job_id, count=skipped)
        if shifted is not None:
            log.info("jobs.dst_shifted", job=job_id, slot=shifted)
        return True

    def reschedule_job(
        self, job_id: str, *, every_seconds: int, next_run_at: float, schedule: Schedule
    ) -> bool:
        """Change the schedule and the next run together, without touching the outbox.

        A full-row write from a snapshot taken before the runner queued or
        cleared a report would hide a pending send, or put one back after it
        had already gone out. The schedule columns and ``every_seconds`` move
        in one statement so the row never repeats by one rule and steps by
        another.
        """
        return self._touch_job(
            "UPDATE jobs SET every_seconds = ?, next_run_at = ?, zone = ?, anchor_at = ?, "
            "every_days = ?, at_local = ? WHERE id = ? AND cancelled_at IS NULL",
            (every_seconds, next_run_at, *_schedule_values(schedule)),
            job_id,
        )

    def backfill_job_schedules(self, zone: str, *, now: float | None = None) -> int:
        """Give every row an older build wrote its wall-clock schedule. Returns how many.

        That is every row with no zone, and every active row whose schedule
        no longer agrees with the ``every_seconds`` and next run beside it.
        This build writes the three together, so a disagreement means a
        version 14 image stepped or rescheduled the row after a rollback.
        A whole-day row whose gap is still its day count was only stepped,
        so it keeps its stored local hour and goes back on the slot its
        stepped run stands for, which stays due when a start lands just
        before that stepped run; any other disagreement is rebuilt from what
        that image wrote. A row
        that still agrees is left alone, so a start does not move a healthy
        row.

        Cancelled rows with no zone are filled too, so a row is never half
        migrated. The slot the person chose is the stored next run for a row
        that never ran, and one gap before it for a row that did: 0.1.8
        stepped a daily row by 86,400 seconds, so across a daylight saving
        change its stored next run is an hour off and only the slot before
        it holds the hour they named. A row that never ran keeps its next
        run, even when it is already past, because nothing has moved it. A
        whole-day row that ran drops the drifted instant for its next local
        slot, so it does not text at the wrong hour and again at the right
        one the same morning. Any other row keeps its next run. A row that
        cannot be read is logged and left for the next start rather than
        stopping the gateway.
        """
        moment = time.time() if now is None else now
        household = ZoneInfo(zone)
        migrated: list[tuple[str, str, int | None, str | None, bool]] = []
        try:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE zone IS NULL OR cancelled_at IS NULL"
                ).fetchall()
                for row in rows:
                    rebuilt = row["zone"] is not None
                    filled: tuple[Schedule, float] | None = None
                    if rebuilt:
                        try:
                            job = _job_from_row(row)
                        except StoreError:
                            log.warning("store.job_schedule_unreadable", job=row["id"])
                            continue
                        if _schedule_holds(job):
                            continue
                        filled = _realigned(job)
                    if filled is None:
                        filled = _legacy_schedule(row, household, now=moment)
                    if filled is None:
                        log.warning("store.job_schedule_unreadable", job=row["id"])
                        continue
                    schedule, next_run_at = filled
                    values = (*_schedule_values(schedule), next_run_at)
                    if rebuilt and values == _stored_values(row):
                        continue
                    self._conn.execute(
                        "UPDATE jobs SET zone = ?, anchor_at = ?, every_days = ?, at_local = ?, "
                        "next_run_at = ? WHERE id = ?",
                        (*values, row["id"]),
                    )
                    migrated.append(
                        (
                            "store.job_schedule_rebuilt"
                            if rebuilt
                            else "store.job_schedule_migrated",
                            str(row["id"]),
                            schedule.every_days,
                            schedule.at_local,
                            row["cancelled_at"] is not None,
                        )
                    )
        except sqlite3.Error as exc:
            raise StoreError("job schedules could not be migrated") from exc
        # Logged after the commit so a rolled-back fill is not reported as done.
        for event, job_id, every_days, at_local, cancelled in migrated:
            log.info(
                event,
                job=job_id,
                every_days=every_days,
                at_local=at_local,
                cancelled=cancelled,
            )
        return len(migrated)

    def active_jobs(self) -> list[Job]:
        """Every agent's active jobs, soonest first. For the operator, not the model."""
        try:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE cancelled_at IS NULL ORDER BY next_run_at ASC, id ASC"
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError("jobs are unreadable") from exc
        return _jobs_from_rows(rows)

    def mark_relay_sent(self, job_id: str) -> bool:
        """The other phone already has this send. A retry must not deliver it again."""
        return self._touch_job(
            "UPDATE jobs SET relay_sent = 1 WHERE id = ? AND cancelled_at IS NULL",
            (),
            job_id,
        )

    def mark_relay_recorded(self, job_id: str) -> bool:
        """The recipient's assistant has the relay record. A retry must not write it again."""
        return self._touch_job(
            "UPDATE jobs SET relay_recorded = 1 WHERE id = ? AND cancelled_at IS NULL",
            (),
            job_id,
        )

    def clear_job_outbox(self, job_id: str) -> None:
        """Drop a pending report. Used after persist when the job is already cancelled."""
        try:
            with self._conn:
                self._conn.execute(
                    "UPDATE jobs SET pending_text = NULL, pending_untrusted = 0, "
                    "pending_after = NULL, pending_delivered = 0 WHERE id = ?",
                    (job_id,),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"job {job_id} could not be saved") from exc

    def _touch_job(self, sql: str, values: tuple[Any, ...], job_id: str) -> bool:
        try:
            with self._conn:
                cursor = self._conn.execute(sql, (*values, job_id))
        except sqlite3.Error as exc:
            raise StoreError(f"job {job_id} could not be saved") from exc
        return cursor.rowcount > 0

    def cancel_job(self, job_id: str, *, at: float) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    "UPDATE jobs SET cancelled_at = ? WHERE id = ? AND cancelled_at IS NULL",
                    (at, job_id),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"job {job_id} could not be saved") from exc

    def gmail_token(self, agent: str) -> tuple[str, str, float] | None:
        """Refresh token, access token, and expiry. None when this agent has not signed in."""
        try:
            row = self._conn.execute(
                "SELECT refresh_token, access_token, expires_at FROM gmail_tokens WHERE agent = ?",
                (agent,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"gmail token for {agent} is unreadable") from exc
        if row is None:
            return None
        refresh = row["refresh_token"]
        access = row["access_token"] if isinstance(row["access_token"], str) else ""
        if not isinstance(refresh, str) or not refresh:
            return None
        return refresh, access, float(row["expires_at"] or 0)

    def save_gmail_token(
        self,
        agent: str,
        *,
        refresh_token: str,
        access_token: str,
        expires_at: float,
    ) -> None:
        if not refresh_token:
            raise StoreError("gmail refresh token is empty")
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO gmail_tokens (agent, refresh_token, access_token, expires_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(agent) DO UPDATE SET "
                    "refresh_token = excluded.refresh_token, "
                    "access_token = excluded.access_token, "
                    "expires_at = excluded.expires_at",
                    (agent, refresh_token, access_token, expires_at),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"gmail token for {agent} could not be saved") from exc

    def last_update_report(self) -> str:
        """The last update summary sent to the operator, or empty."""
        try:
            row = self._conn.execute("SELECT report FROM update_notices WHERE id = 1").fetchone()
        except sqlite3.Error as exc:
            raise StoreError("update notice is unreadable") from exc
        if row is None:
            return ""
        return str(row["report"])

    def save_update_report(self, report: str, now: float) -> None:
        """Remember a summary so the same stale pin is not texted again."""
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO update_notices (id, report, sent_at) VALUES (1, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "report = excluded.report, sent_at = excluded.sent_at",
                    (report, now),
                )
        except sqlite3.Error as exc:
            raise StoreError("update notice could not be saved") from exc

    def shared_snapshot(self) -> list[tuple[str, str | None, list[tuple[str, bool]]]]:
        """Every list, its owner (none when shared), and its items."""
        try:
            lists = self._conn.execute(
                "SELECT id, name, owner FROM shared_lists ORDER BY updated_at, name"
            ).fetchall()
            items = self._conn.execute(
                "SELECT list_id, text, done FROM shared_items ORDER BY updated_at, id"
            ).fetchall()
        except sqlite3.Error as exc:
            raise StoreError("shared lists are unreadable") from exc
        by_list: dict[str, list[tuple[str, bool]]] = {row["id"]: [] for row in lists}
        for row in items:
            bucket = by_list.get(row["list_id"])
            if bucket is not None:
                bucket.append((row["text"], bool(row["done"])))
        found: list[tuple[str, str | None, list[tuple[str, bool]]]] = []
        for row in lists:
            owner = row["owner"] if isinstance(row["owner"], str) else None
            found.append((row["name"], owner, by_list[row["id"]]))
        return found

    def apply_shared(
        self,
        *,
        name: str,
        add: tuple[str, ...],
        done: tuple[str, ...],
        remove: tuple[str, ...],
        actor: str,
        owner: str | None,
        share: bool,
        now: float,
    ) -> None:
        """Apply one confirmed list change. A missing item changes nothing."""
        try:
            with self._conn:
                if share:
                    self._share_private_list(name, actor, now)
                    return
                row = self._find_list(name, owner)
                if row is None:
                    if done or remove or not add:
                        raise StoreError(f"there is no list named {name}")
                    list_id = uuid.uuid4().hex
                    self._conn.execute(
                        "INSERT INTO shared_lists (id, name, owner, created_by, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (list_id, name, owner, actor, now),
                    )
                else:
                    list_id = row["id"]
                existing = self._conn.execute(
                    "SELECT id, text, done FROM shared_items WHERE list_id = ?",
                    (list_id,),
                ).fetchall()
                by_text: dict[str, list[sqlite3.Row]] = {}
                for item in existing:
                    by_text.setdefault(str(item["text"]).casefold(), []).append(item)
                for text in done:
                    open_items = by_text.get(text.casefold(), [])
                    match = next((item for item in open_items if not item["done"]), None)
                    if match is None:
                        raise StoreError(f"{text!r} is not an open item on {name}")
                for text in remove:
                    if text.casefold() not in by_text:
                        raise StoreError(f"{text!r} is not on {name}")
                for text in add:
                    if any(not item["done"] for item in by_text.get(text.casefold(), [])):
                        raise StoreError(f"{text!r} is already on {name}")
                for text in remove:
                    for item in by_text[text.casefold()]:
                        self._conn.execute("DELETE FROM shared_items WHERE id = ?", (item["id"],))
                for text in done:
                    match = next(item for item in by_text[text.casefold()] if not item["done"])
                    self._conn.execute(
                        "UPDATE shared_items SET done = 1, updated_at = ? WHERE id = ?",
                        (now, match["id"]),
                    )
                for text in add:
                    self._conn.execute(
                        "INSERT INTO shared_items "
                        "(id, list_id, text, done, created_by, updated_at) "
                        "VALUES (?, ?, ?, 0, ?, ?)",
                        (uuid.uuid4().hex, list_id, text, actor, now),
                    )
                self._conn.execute(
                    "UPDATE shared_lists SET updated_at = ? WHERE id = ?",
                    (now, list_id),
                )
        except StoreError:
            raise
        except sqlite3.Error as exc:
            raise StoreError(f"shared list {name} could not be saved") from exc

    def _init_schema(self) -> None:
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise StoreError(
                f"state database is newer than this build (version {version} > {SCHEMA_VERSION})"
            )
        if version == SCHEMA_VERSION:
            return
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._ensure_job_pending_columns()
            self._ensure_job_reminder_columns()
            self._ensure_job_relay_column()
            self._ensure_job_relay_sent_column()
            self._ensure_job_relay_recorded_column()
            self._ensure_job_tainted_column()
            self._ensure_job_schedule_columns()
            self._ensure_shared_list_owner()
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _ensure_job_pending_columns(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if not cols:
            return
        additions = (
            ("pending_text", "TEXT"),
            ("pending_untrusted", "INTEGER NOT NULL DEFAULT 0"),
            ("pending_after", "TEXT"),
            ("pending_delivered", "INTEGER NOT NULL DEFAULT 0"),
        )
        for name, decl in additions:
            if name not in cols:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")

    def _ensure_job_reminder_columns(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if not cols:
            return
        additions = (
            ("once", "INTEGER NOT NULL DEFAULT 0"),
            ("verbatim", "INTEGER NOT NULL DEFAULT 0"),
        )
        for name, decl in additions:
            if name not in cols:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")

    def _ensure_job_relay_column(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if not cols or "relay" in cols:
            return
        self._conn.execute("ALTER TABLE jobs ADD COLUMN relay INTEGER NOT NULL DEFAULT 0")

    def _ensure_job_relay_sent_column(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if not cols or "relay_sent" in cols:
            return
        self._conn.execute("ALTER TABLE jobs ADD COLUMN relay_sent INTEGER NOT NULL DEFAULT 0")

    def _ensure_job_relay_recorded_column(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if not cols or "relay_recorded" in cols:
            return
        self._conn.execute("ALTER TABLE jobs ADD COLUMN relay_recorded INTEGER NOT NULL DEFAULT 0")

    def _ensure_job_tainted_column(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if not cols or "tainted" in cols:
            return
        # Rows saved before this column were treated as trusted; keep that.
        self._conn.execute("ALTER TABLE jobs ADD COLUMN tainted INTEGER NOT NULL DEFAULT 0")

    def _ensure_job_schedule_columns(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if not cols:
            return
        # Nullable, so a version 14 build can still write rows after a
        # downgrade; backfill_job_schedules fills them on the next start.
        for name, decl in _SCHEDULE_COLUMNS:
            if name not in cols:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")

    def _ensure_shared_list_owner(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(shared_lists)")}
        if not cols:
            return
        if "owner" not in cols:
            self._conn.execute("ALTER TABLE shared_lists ADD COLUMN owner TEXT")
        self._conn.execute("DROP INDEX IF EXISTS shared_lists_name")
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS shared_lists_shared_name "
            "ON shared_lists(lower(name)) WHERE owner IS NULL"
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS shared_lists_private_name "
            "ON shared_lists(owner, lower(name)) WHERE owner IS NOT NULL"
        )

    def _find_list(self, name: str, owner: str | None) -> sqlite3.Row | None:
        if owner is None:
            found = self._conn.execute(
                "SELECT id FROM shared_lists WHERE lower(name) = lower(?) AND owner IS NULL",
                (name,),
            ).fetchone()
        else:
            found = self._conn.execute(
                "SELECT id FROM shared_lists WHERE lower(name) = lower(?) AND owner = ?",
                (name, owner),
            ).fetchone()
        if found is None:
            return None
        return cast(sqlite3.Row, found)

    def _share_private_list(self, name: str, actor: str, now: float) -> None:
        row = self._find_list(name, actor)
        if row is None:
            raise StoreError(f"there is no list named {name}")
        taken = self._find_list(name, None)
        if taken is not None:
            raise StoreError(f"a shared list named {name} already exists")
        self._conn.execute(
            "UPDATE shared_lists SET owner = NULL, updated_at = ? WHERE id = ?",
            (now, row["id"]),
        )

    def _replace_history(self, agent: str, persisted: list[Message]) -> None:
        self._conn.execute("DELETE FROM messages WHERE agent = ?", (agent,))
        self._conn.executemany(
            "INSERT INTO messages "
            "(agent, seq, role, content, tool_calls, tool_call_id, untrusted, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    agent,
                    seq,
                    message.role,
                    message.content,
                    _tool_calls_json(message),
                    message.tool_call_id,
                    1 if message.untrusted else 0,
                    message.created_at if message.created_at is not None else time.time(),
                )
                for seq, message in enumerate(persisted)
            ],
        )

    def _replace_proposal_row(self, agent: str, proposal: Proposal | None) -> None:
        self._conn.execute("DELETE FROM proposals WHERE agent = ?", (agent,))
        if proposal is None:
            return
        self._conn.execute(
            "INSERT INTO proposals (agent, calls, tainted, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                agent,
                _proposal_calls_json(proposal.calls),
                1 if proposal.tainted else 0,
                proposal.created_at,
                proposal.expires_at,
            ),
        )


def _tool_calls_json(message: Message) -> str | None:
    if not message.tool_calls:
        return None
    payload: list[dict[str, str]] = [
        {"id": call.id, "name": call.name, "arguments": call.arguments}
        for call in message.tool_calls
    ]
    return json.dumps(payload)


def _message_from_row(row: sqlite3.Row, *, agent: str) -> Message:
    role = row["role"]
    if role not in _PERSISTED_ROLES:
        raise StoreError(f"history for {agent} is unreadable")
    raw_calls = row["tool_calls"]
    tool_calls = _parse_tool_calls(raw_calls, agent=agent, what="history") if raw_calls else []
    created_at = row["created_at"]
    if not isinstance(created_at, (int, float)):
        raise StoreError(f"history for {agent} is unreadable")
    return Message(
        role=role,
        content=row["content"],
        tool_calls=tool_calls,
        tool_call_id=row["tool_call_id"],
        untrusted=bool(row["untrusted"]),
        created_at=float(created_at),
    )


def _parse_tool_calls(raw: str, *, agent: str, what: str = "history") -> list[ToolCall]:
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StoreError(f"{what} for {agent} is unreadable") from exc
    if not isinstance(parsed, list):
        raise StoreError(f"{what} for {agent} is unreadable")
    calls: list[ToolCall] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise StoreError(f"{what} for {agent} is unreadable")
        call_id = item.get("id")
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise StoreError(f"{what} for {agent} is unreadable")
        if not isinstance(arguments, str):
            raise StoreError(f"{what} for {agent} is unreadable")
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return calls


def _proposal_calls_json(calls: tuple[ToolCall, ...] | list[ToolCall]) -> str:
    payload: list[dict[str, str]] = [
        {"id": call.id, "name": call.name, "arguments": call.arguments} for call in calls
    ]
    return json.dumps(payload)


def _proposal_calls(raw: object, *, agent: str) -> list[ToolCall]:
    if not isinstance(raw, str):
        raise StoreError(f"proposal for {agent} is unreadable")
    return _parse_tool_calls(raw, agent=agent, what="proposal")


def _job_from_row(row: sqlite3.Row) -> Job:
    kind = row["kind"]
    if kind not in ("schedule", "watch"):
        raise StoreError(f"job {row['id']} is unreadable")
    typed_kind: JobKind = kind
    every_seconds = row["every_seconds"]
    if not isinstance(every_seconds, int) or every_seconds <= 0:
        raise StoreError(f"job {row['id']} is unreadable")
    created_at = row["created_at"]
    next_run_at = row["next_run_at"]
    if not isinstance(created_at, (int, float)) or not isinstance(next_run_at, (int, float)):
        raise StoreError(f"job {row['id']} is unreadable")
    ttl_seconds = row["ttl_seconds"]
    if ttl_seconds is not None and not isinstance(ttl_seconds, int):
        raise StoreError(f"job {row['id']} is unreadable")
    expires_at = row["expires_at"]
    if expires_at is not None and not isinstance(expires_at, (int, float)):
        raise StoreError(f"job {row['id']} is unreadable")
    last_run_at = row["last_run_at"]
    if last_run_at is not None and not isinstance(last_run_at, (int, float)):
        raise StoreError(f"job {row['id']} is unreadable")
    cancelled_at = row["cancelled_at"]
    if cancelled_at is not None and not isinstance(cancelled_at, (int, float)):
        raise StoreError(f"job {row['id']} is unreadable")
    name = row["name"]
    prompt = row["prompt"]
    agent = row["agent"]
    job_id = row["id"]
    if not isinstance(name, str) or not isinstance(prompt, str):
        raise StoreError(f"job {job_id} is unreadable")
    if not isinstance(agent, str) or not isinstance(job_id, str):
        raise StoreError("job row is unreadable")
    return Job(
        id=job_id,
        agent=agent,
        kind=typed_kind,
        name=name,
        prompt=prompt,
        every_seconds=every_seconds,
        ttl_seconds=ttl_seconds,
        created_at=float(created_at),
        expires_at=float(expires_at) if expires_at is not None else None,
        next_run_at=float(next_run_at),
        last_run_at=float(last_run_at) if last_run_at is not None else None,
        cancelled_at=float(cancelled_at) if cancelled_at is not None else None,
        pending_text=_optional_str(row["pending_text"], job_id),
        pending_untrusted=_flag(row["pending_untrusted"], job_id),
        pending_after=_pending_after(row["pending_after"], job_id),
        pending_delivered=_flag(row["pending_delivered"], job_id),
        once=_flag(row["once"], job_id) if "once" in row.keys() else False,
        verbatim=_flag(row["verbatim"], job_id) if "verbatim" in row.keys() else False,
        relay=_flag(row["relay"], job_id) if "relay" in row.keys() else False,
        relay_sent=_flag(row["relay_sent"], job_id) if "relay_sent" in row.keys() else False,
        relay_recorded=(
            _flag(row["relay_recorded"], job_id) if "relay_recorded" in row.keys() else False
        ),
        tainted=_flag(row["tainted"], job_id) if "tainted" in row.keys() else False,
        zone=_optional_str(_column(row, "zone"), job_id),
        anchor_at=_optional_float(_column(row, "anchor_at"), job_id),
        every_days=_optional_days(_column(row, "every_days"), job_id),
        at_local=_optional_str(_column(row, "at_local"), job_id),
    )


def _jobs_from_rows(rows: list[sqlite3.Row]) -> list[Job]:
    """Skip a corrupt row so one bad job cannot hide the rest of the list."""
    jobs: list[Job] = []
    for row in rows:
        try:
            jobs.append(_job_from_row(row))
        except StoreError:
            ident = row["id"] if "id" in row.keys() else None
            log.exception("store.job_unreadable", job=ident)
    return jobs


def _optional_str(raw: object, job_id: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise StoreError(f"job {job_id} is unreadable")
    return raw


def _column(row: sqlite3.Row, name: str) -> object:
    """A column an older file may not have yet reads as NULL."""
    return row[name] if name in row.keys() else None


def _optional_float(raw: object, job_id: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise StoreError(f"job {job_id} is unreadable")
    return float(raw)


def _optional_days(raw: object, job_id: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise StoreError(f"job {job_id} is unreadable")
    return raw


def _stored_schedule(job: Job) -> Schedule:
    """The row's schedule. A row an older build wrote has none until the next start fills it."""
    schedule = job.schedule()
    if schedule is None:
        raise StoreError(f"job {job.id} has no schedule yet")
    return schedule


def _schedule_values(schedule: Schedule) -> tuple[str, float, int | None, str | None]:
    return schedule.zone.key, schedule.anchor_at, schedule.every_days, schedule.at_local


def _schedule_holds(job: Job) -> bool:
    """True when a row's schedule still describes its ``every_seconds`` and next run.

    Every write this build makes keeps the stored next run on the schedule:
    the first run is the anchor, and every later one is a slot. A version
    14 image writes only ``every_seconds`` and ``next_run_at``, so after a
    rollback its step or reschedule shows up here as a gap that is not the
    day count, or a next run that is off the grid. A schedule that cannot
    be read does not hold either, so it is rebuilt.
    """
    try:
        schedule = job.schedule()
    except JobError:
        return False
    if schedule is None:
        return False
    if schedule.every_days is not None and job.every_seconds != schedule.every_days * DAY_SECONDS:
        return False
    if job.next_run_at == schedule.anchor_at:
        return True
    if not schedule.repeats:
        return False
    try:
        return next_slot(schedule, after=math.nextafter(job.next_run_at, -math.inf)) == (
            job.next_run_at
        )
    except JobError:
        return False


def _realigned(job: Job) -> tuple[Schedule, float] | None:
    """The stored whole-day schedule and the slot its stepped next run stands for.

    A version 14 image that stepped the row by ``every_seconds`` moved the
    next run off the hour after a daylight saving change or a spring-gap
    run, and the instant one gap back is that shifted hour, not the one the
    person named. The stored ``at_local`` is the named hour, so it is kept.
    The slot is the first one after the run that image stepped from, not
    the first after the start: its next run is a few seconds, or a spring
    hour, past that slot, and a start in between must not push a pending
    run to the next day. A slot already past is due, so the runner sends
    it once and puts the row back on its schedule. None when the gap no
    longer matches the day count, because then the old image rescheduled
    the row and its columns say what runs.
    """
    try:
        schedule = job.schedule()
    except JobError:
        return None
    if schedule is None or schedule.every_days is None:
        return None
    if job.every_seconds != schedule.every_days * DAY_SECONDS:
        return None
    stepped_from = job.next_run_at - job.every_seconds
    try:
        return schedule, next_slot(schedule, after=stepped_from)
    except JobError:
        return None


def _stored_values(row: sqlite3.Row) -> tuple[object, ...]:
    return (row["zone"], row["anchor_at"], row["every_days"], row["at_local"], row["next_run_at"])


def _legacy_schedule(
    row: sqlite3.Row, zone: ZoneInfo, *, now: float
) -> tuple[Schedule, float] | None:
    """The schedule and next run an older build's row stands for, or None if unreadable."""
    every = row["every_seconds"]
    stored = row["next_run_at"]
    last_run_at = row["last_run_at"]
    if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
        return None
    if isinstance(stored, bool) or not isinstance(stored, (int, float)):
        return None
    if last_run_at is not None and not isinstance(last_run_at, (int, float)):
        return None
    ran = last_run_at is not None
    chosen = float(stored) - every if ran else float(stored)
    try:
        schedule = schedule_from_seconds(
            zone=zone, first_run=chosen, every_seconds=every, once=row["once"] == 1
        )
        next_run_at = float(stored)
        if ran and schedule.every_days is not None:
            next_run_at = next_slot(schedule, after=max(now, chosen))
    except (JobError, OverflowError, OSError, ValueError):
        return None
    return schedule, next_run_at


def downgrade(path: Path, *, to: int) -> None:
    """Hand the state file back to the build one schema older, or refuse and change nothing.

    The file must be at ``to + 1``, and the step for ``to`` must recognise
    it. A file it does not recognise may hold a change the older build would
    misread, so nothing is written. The check and the version write happen
    under one lock, so the gateway cannot change the file between them.
    """
    step = _DOWNGRADE_STEPS.get(to)
    if step is None:
        allowed = ", ".join(str(target) for target in sorted(_DOWNGRADE_STEPS))
        raise StoreError(f"cannot downgrade to version {to}; this build can go to {allowed}")
    if not path.is_file():
        raise StoreError(f"no state database at {path}")
    try:
        conn = sqlite3.connect(
            path.resolve().as_uri() + "?mode=rw", uri=True, isolation_level=None, timeout=5.0
        )
    except sqlite3.Error as exc:
        raise StoreError(f"state database unreadable: {path}") from exc
    try:
        conn.execute("BEGIN IMMEDIATE")
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version != to + 1:
            raise StoreError(
                f"state database is version {version}, and the step to {to} starts "
                f"from {to + 1}; refusing to downgrade"
            )
        step(conn)
        conn.execute(f"PRAGMA user_version = {to}")
        conn.execute("COMMIT")
    except sqlite3.Error as exc:
        raise StoreError(f"state database unreadable: {path}") from exc
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        conn.close()
    log.info("store.downgraded", version=to, previous=to + 1)


def _down_to_14(conn: sqlite3.Connection) -> None:
    """Version 15 only added nullable job columns no version 14 query names.

    So the file goes back unchanged, but only if it is exactly what this
    build writes: the same tables, columns, and indexes as a fresh file.
    The shapes are compared through SQLite's own description of each table,
    not the CREATE text, because a migrated file and a fresh one say the
    same thing in different words. When the schema moves past 15, the
    version 15 shape this compares against has to be frozen here.
    """
    differs = _shape_differences(_shape(conn), _expected_shape())
    if differs:
        raise StoreError(
            f"state database does not match schema 15 ({', '.join(differs)} differ); "
            "refusing to downgrade"
        )


# Each step hands a file at ``to + 1`` back to ``to``. A later schema change
# that keeps the previous image working adds its own.
_DOWNGRADE_STEPS: dict[int, Callable[[sqlite3.Connection], None]] = {14: _down_to_14}


_Shape = dict[str, object]


def _shape(conn: sqlite3.Connection) -> _Shape:
    """Tables, columns, indexes, and foreign keys as SQLite reports them."""
    shape: _Shape = {}
    objects = conn.execute(
        "SELECT type, name FROM sqlite_master WHERE substr(name, 1, 7) != 'sqlite_'"
    ).fetchall()
    for kind, name in objects:
        if kind == "index":
            # Read below through the table it belongs to.
            continue
        if kind != "table":
            shape[f"{kind} {name}"] = kind
            continue
        columns = frozenset(
            (col[1], str(col[2]).upper(), col[3], col[4], col[5])
            for col in conn.execute("SELECT * FROM pragma_table_info(?)", (name,))
        )
        indexes = frozenset(
            (
                index[1],
                index[2],
                index[3],
                index[4],
                tuple(
                    (key[1], key[2], key[3], key[4])
                    for key in conn.execute("SELECT * FROM pragma_index_xinfo(?)", (index[1],))
                    if key[5]
                ),
            )
            for index in conn.execute("SELECT * FROM pragma_index_list(?)", (name,))
        )
        foreign = frozenset(
            tuple(key[2:])
            for key in conn.execute("SELECT * FROM pragma_foreign_key_list(?)", (name,))
        )
        shape[f"table {name}"] = (columns, indexes, foreign)
    return shape


def _expected_shape() -> _Shape:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(_SCHEMA)
        return _shape(conn)
    finally:
        conn.close()


def _shape_differences(actual: _Shape, expected: _Shape) -> list[str]:
    return sorted(
        name for name in actual.keys() | expected.keys() if actual.get(name) != expected.get(name)
    )


def _flag(raw: object, job_id: str) -> bool:
    if raw in (0, 1):
        return bool(raw)
    if raw is None:
        return False
    raise StoreError(f"job {job_id} is unreadable")


def _pending_after(raw: object, job_id: str) -> Literal["advance", "cancel"] | None:
    if raw is None:
        return None
    if raw in ("advance", "cancel"):
        return raw
    raise StoreError(f"job {job_id} is unreadable")
