"""Durable household state. One SQLite file in the gateway state volume.

History, untrusted labels, and the pairing allowlist live here so a reboot
cannot forget a conversation or silently clear taint. Staging and jobs will
join this file later; they are not in this schema yet.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import structlog

from assistai.errors import StoreError
from assistai.inference.types import Message, ToolCall
from assistai.signal.numbers import InvalidNumberError, normalize_e164

log = structlog.get_logger(__name__)

SCHEMA_VERSION = 1
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
        persisted = [message for message in messages if message.role in _PERSISTED_ROLES]
        try:
            with self._conn:
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
        except sqlite3.Error as exc:
            raise StoreError(f"history for {agent} could not be saved") from exc

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
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


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
    tool_calls = _parse_tool_calls(raw_calls, agent=agent) if raw_calls else []
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


def _parse_tool_calls(raw: str, *, agent: str) -> list[ToolCall]:
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StoreError(f"history for {agent} is unreadable") from exc
    if not isinstance(parsed, list):
        raise StoreError(f"history for {agent} is unreadable")
    calls: list[ToolCall] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise StoreError(f"history for {agent} is unreadable")
        call_id = item.get("id")
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise StoreError(f"history for {agent} is unreadable")
        if not isinstance(arguments, str):
            raise StoreError(f"history for {agent} is unreadable")
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return calls
