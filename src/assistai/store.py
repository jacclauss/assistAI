"""Durable household state. One SQLite file in the gateway state volume.

History, untrusted labels, the pairing allowlist, staged proposals, jobs,
and per-person Gmail refresh tokens live here so a reboot cannot forget a
conversation, silently clear taint, skip a confirmation, or drop a schedule.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Literal, cast

import structlog

from assistai.errors import StoreError
from assistai.inference.types import Message, ToolCall
from assistai.jobs import Job, JobKind
from assistai.signal.numbers import InvalidNumberError, normalize_e164
from assistai.staging import Proposal

log = structlog.get_logger(__name__)

SCHEMA_VERSION = 8
_DB_NAME = "assistai.sqlite"

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
    pending_delivered INTEGER NOT NULL DEFAULT 0
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
"""

_PERSISTED_ROLES = frozenset({"user", "assistant", "tool"})


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

        An expired row is deleted so a later yes cannot fire it. Corrupt JSON
        fails closed: executing garbage would be worse than asking again.
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
        if not isinstance(expires_at, (int, float)):
            raise StoreError(f"proposal for {agent} is unreadable")
        if float(expires_at) <= (now if now is not None else time.time()):
            self.clear_proposal(agent)
            return None, True
        created_at = row["created_at"]
        if not isinstance(created_at, (int, float)):
            raise StoreError(f"proposal for {agent} is unreadable")
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
                    "pending_delivered) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        """Schedule the next run and clear the outbox. False if cancelled meanwhile.

        The interval comes from the stored row, not from the runner's copy, so a
        reschedule confirmed during the run is what takes effect.
        """
        return self._touch_job(
            "UPDATE jobs SET next_run_at = ? + every_seconds, last_run_at = ?, "
            "pending_text = NULL, pending_untrusted = 0, pending_after = NULL, "
            "pending_delivered = 0 WHERE id = ? AND cancelled_at IS NULL",
            (now, now),
            job_id,
        )

    def reschedule_job(self, job_id: str, *, every_seconds: int, next_run_at: float) -> bool:
        """Change the interval without touching the outbox.

        A full-row write from a snapshot taken before the runner queued or
        cleared a report would hide a pending send, or put one back after it
        had already gone out.
        """
        return self._touch_job(
            "UPDATE jobs SET every_seconds = ?, next_run_at = ? "
            "WHERE id = ? AND cancelled_at IS NULL",
            (every_seconds, next_run_at),
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
                    match = next(
                        item for item in by_text[text.casefold()] if not item["done"]
                    )
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
