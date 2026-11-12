from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from structlog.testing import capture_logs

from assistai.clock import Schedule
from assistai.errors import StoreError
from assistai.inference.types import Message, ToolCall
from assistai.jobs import Job
from assistai.staging import Proposal
from assistai.store import SCHEMA_VERSION, Store, downgrade
from tests.clock_fakes import CHICAGO, chicago, wall

# What job_reschedule stores for "every 600 seconds" when the next run is at 610.
_ten_minutes = Schedule(zone=CHICAGO, anchor_at=610.0, every_seconds=600)


def _store(tmp_path: Path) -> Store:
    return Store(tmp_path / "assistai.sqlite")


def test_history_round_trips_untrusted_and_tool_calls(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = [
        Message(role="system", content="you are jacob"),
        Message(role="user", content="look this up", created_at=1.0),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="web_fetch", arguments="{}")],
            created_at=2.0,
        ),
        Message(
            role="tool",
            content="injected",
            tool_call_id="c1",
            untrusted=True,
            created_at=3.0,
        ),
        Message(role="assistant", content="here it is", untrusted=True, created_at=4.0),
    ]

    store.save_history("jacob", original)
    loaded = store.load_history("jacob")

    assert [message.role for message in loaded] == ["user", "assistant", "tool", "assistant"]
    assert loaded[2].untrusted is True
    assert loaded[3].untrusted is True
    assert loaded[1].tool_calls[0].name == "web_fetch"
    assert loaded[2].tool_call_id == "c1"
    store.close()


def test_history_is_isolated_per_agent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_history("jacob", [Message(role="user", content="mine", created_at=1.0)])
    store.save_history("spouse", [Message(role="user", content="hers", created_at=1.0)])

    assert store.load_history("jacob")[0].content == "mine"
    assert store.load_history("spouse")[0].content == "hers"
    store.close()


def test_save_replaces_previous_history(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_history("jacob", [Message(role="user", content="old", created_at=1.0)])
    store.save_history("jacob", [Message(role="user", content="new", created_at=2.0)])

    loaded = store.load_history("jacob")
    assert [message.content for message in loaded] == ["new"]
    store.close()


def test_allowlist_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "assistai.sqlite"
    store = Store(path)
    store.admit("+1 (555) 555-0102")
    store.close()

    reopened = Store(path)
    assert reopened.approved() == frozenset({"+15555550102"})
    reopened.close()


def test_legacy_json_allowlist_is_imported_once(tmp_path: Path) -> None:
    legacy = tmp_path / "signal-allowlist.json"
    legacy.write_text(
        json.dumps({"approved": ["+15555550102", "not-a-number"]}) + "\n", encoding="utf-8"
    )
    store = _store(tmp_path)

    store.import_legacy_allowlist(legacy)

    assert store.approved() == frozenset({"+15555550102"})
    assert not legacy.is_file()
    assert (tmp_path / "signal-allowlist.json.migrated").is_file()

    store.import_legacy_allowlist(legacy)
    assert store.approved() == frozenset({"+15555550102"})
    store.close()


def test_legacy_json_is_kept_when_admit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-migrated file must stay in place so the next boot can retry."""
    legacy = tmp_path / "signal-allowlist.json"
    legacy.write_text(json.dumps({"approved": ["+15555550102"]}) + "\n", encoding="utf-8")
    store = _store(tmp_path)

    def boom(_number: str) -> None:
        raise StoreError("disk full")

    monkeypatch.setattr(store, "admit", boom)
    store.import_legacy_allowlist(legacy)

    assert legacy.is_file()
    assert not (tmp_path / "signal-allowlist.json.migrated").is_file()
    store.close()


def test_corrupt_history_fails_closed(tmp_path: Path) -> None:
    """An empty load would launder taint. Refuse instead."""
    store = _store(tmp_path)
    store.save_history("jacob", [Message(role="user", content="hi", created_at=1.0)])
    store.close()

    conn = sqlite3.connect(tmp_path / "assistai.sqlite")
    conn.execute("UPDATE messages SET role = 'wizard' WHERE agent = 'jacob'")
    conn.commit()
    conn.close()

    reopened = Store(tmp_path / "assistai.sqlite")
    with pytest.raises(StoreError, match="unreadable"):
        reopened.load_history("jacob")
    reopened.close()


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "assistai.sqlite"
    Store(path).close()
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    with pytest.raises(StoreError, match="newer than this build"):
        Store(path)


def test_proposal_round_trips_and_replaces(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _proposal("jacob", '{"id": "1"}', expires_at=9_999_999_999)
    second = _proposal("jacob", '{"id": "2"}', expires_at=9_999_999_999)

    store.replace_proposal(first)
    store.replace_proposal(second)

    loaded = store.load_proposal("jacob")
    assert loaded is not None
    assert loaded.calls[0].arguments == '{"id": "2"}'
    assert store.load_proposal("spouse") is None
    store.close()


def test_expired_proposal_cannot_be_loaded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.replace_proposal(_proposal("jacob", "{}", created_at=1.0, expires_at=2.0))

    assert store.load_proposal("jacob", now=3.0) is None
    assert store.load_proposal("jacob", now=3.0) is None
    store.close()


def test_take_proposal_reports_expiry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.replace_proposal(_proposal("jacob", "{}", created_at=1.0, expires_at=2.0))

    live, expired = store.take_proposal("jacob", now=3.0)
    assert live is None
    assert expired is True
    missing, expired_again = store.take_proposal("jacob", now=3.0)
    assert missing is None
    assert expired_again is False
    store.close()


def test_corrupt_proposal_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.replace_proposal(_proposal("jacob", "{}", expires_at=9_999_999_999))
    store.close()

    conn = sqlite3.connect(tmp_path / "assistai.sqlite")
    conn.execute("UPDATE proposals SET calls = 'not-json' WHERE agent = 'jacob'")
    conn.commit()
    conn.close()

    reopened = Store(tmp_path / "assistai.sqlite")
    with pytest.raises(StoreError, match="unreadable"):
        reopened.load_proposal("jacob")
    reopened.close()


def test_schema_v1_gains_a_proposals_table(tmp_path: Path) -> None:
    path = tmp_path / "assistai.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE messages (
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
        CREATE TABLE allowlist (
            number TEXT PRIMARY KEY NOT NULL,
            admitted_at REAL NOT NULL
        );
        """
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    store = Store(path)
    store.replace_proposal(_proposal("jacob", "{}", expires_at=9_999_999_999))
    assert store.load_proposal("jacob") is not None
    store.close()


def test_jobs_round_trip_and_due_query(tmp_path: Path) -> None:
    from assistai.jobs import Job

    store = _store(tmp_path)
    job = Job(
        id="jabcd1234",
        agent="jacob",
        kind="schedule",
        name="morning email",
        prompt="check mail",
        every_seconds=86400,
        ttl_seconds=None,
        created_at=1.0,
        expires_at=None,
        next_run_at=10.0,
        last_run_at=None,
        cancelled_at=None,
    )
    store.save_job(job)
    store.close()

    reopened = Store(tmp_path / "assistai.sqlite")
    loaded = reopened.find_job("jacob", name="Morning Email")
    assert loaded is not None
    assert loaded.id == "jabcd1234"
    assert loaded.prompt == "check mail"
    assert reopened.due_jobs(9.0) == []
    assert reopened.due_jobs(10.0)[0].id == "jabcd1234"
    reopened.cancel_job("jabcd1234", at=11.0)
    assert reopened.list_jobs("jacob") == []
    reopened.close()


def test_find_job_requires_name_and_id_to_agree(tmp_path: Path) -> None:
    """A cancel preview that names one job must not delete another by id."""
    from assistai.jobs import Job

    store = _store(tmp_path)

    def job(job_id: str, name: str) -> Job:
        return Job(
            id=job_id,
            agent="jacob",
            kind="schedule",
            name=name,
            prompt="check mail",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=10.0,
            last_run_at=None,
            cancelled_at=None,
        )

    store.save_job(job("jabcd1234", "morning email"))
    store.save_job(job("jeeee9999", "evening email"))
    assert store.find_job("jacob", name="morning email", job_id="jeeee9999") is None
    matched = store.find_job("jacob", name="morning email", job_id="jabcd1234")
    assert matched is not None
    assert matched.id == "jabcd1234"
    store.close()


def test_pending_outbox_is_due_even_when_next_run_is_in_the_future(tmp_path: Path) -> None:
    """A reschedule must not hide a report that already ran."""
    from assistai.jobs import Job

    store = _store(tmp_path)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="morning email",
            prompt="check mail",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=9_999.0,
            last_run_at=None,
            cancelled_at=None,
            pending_text="[job: morning email]\nheld",
            pending_after="advance",
        )
    )

    due = store.due_jobs(10.0)
    assert len(due) == 1
    assert due[0].pending_text is not None
    store.close()


def test_reschedule_does_not_restore_a_cleared_outbox(tmp_path: Path) -> None:
    from assistai.jobs import Job

    store = _store(tmp_path)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="morning email",
            prompt="check mail",
            every_seconds=60,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=10.0,
            last_run_at=None,
            cancelled_at=None,
            pending_text="[job: morning email]\nalready sent",
            pending_after="advance",
            pending_delivered=True,
        )
    )
    assert store.advance_job("jabcd1234", now=10.0)
    assert store.reschedule_job(
        "jabcd1234", every_seconds=600, next_run_at=610.0, schedule=_ten_minutes
    )

    loaded = store.find_job("jacob", name="morning email")
    assert loaded is not None
    assert loaded.pending_text is None
    assert loaded.every_seconds == 600
    assert loaded.next_run_at == 610.0
    store.close()


def test_two_active_jobs_cannot_share_a_name(tmp_path: Path) -> None:
    from assistai.jobs import Job

    store = _store(tmp_path)
    store.save_job(
        Job(
            id="jaaaa1111",
            agent="jacob",
            kind="schedule",
            name="morning email",
            prompt="check mail",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=10.0,
            last_run_at=None,
            cancelled_at=None,
        )
    )
    with pytest.raises(StoreError):
        store.save_job(
            Job(
                id="jbbbb2222",
                agent="jacob",
                kind="schedule",
                name="Morning Email",
                prompt="again",
                every_seconds=86400,
                ttl_seconds=None,
                created_at=1.0,
                expires_at=None,
                next_run_at=10.0,
                last_run_at=None,
                cancelled_at=None,
            )
        )
    store.close()


def test_save_job_does_not_overwrite_another_row(tmp_path: Path) -> None:
    """Create is insert-only. Reusing an id must not rename someone else's job."""
    from assistai.jobs import Job

    store = _store(tmp_path)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="morning email",
            prompt="check mail",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=10.0,
            last_run_at=None,
            cancelled_at=None,
        )
    )
    with pytest.raises(StoreError):
        store.save_job(
            Job(
                id="jabcd1234",
                agent="jacob",
                kind="schedule",
                name="evening email",
                prompt="again",
                every_seconds=86400,
                ttl_seconds=None,
                created_at=2.0,
                expires_at=None,
                next_run_at=20.0,
                last_run_at=None,
                cancelled_at=None,
            )
        )
    kept = store.find_job("jacob", name="morning email")
    assert kept is not None
    assert kept.prompt == "check mail"
    assert store.find_job("jacob", name="evening email") is None
    store.close()


def test_schema_v2_gains_a_jobs_table(tmp_path: Path) -> None:
    from assistai.jobs import Job

    path = tmp_path / "assistai.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE messages (
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
        CREATE TABLE allowlist (
            number TEXT PRIMARY KEY NOT NULL,
            admitted_at REAL NOT NULL
        );
        CREATE TABLE proposals (
            agent TEXT PRIMARY KEY NOT NULL,
            calls TEXT NOT NULL,
            tainted INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        );
        """
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    store = Store(path)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="watch",
            name="flights",
            prompt="look",
            every_seconds=60,
            ttl_seconds=3600,
            created_at=1.0,
            expires_at=3601.0,
            next_run_at=61.0,
            last_run_at=None,
            cancelled_at=None,
        )
    )
    assert store.find_job("jacob", name="flights") is not None
    store.close()


def test_schema_v4_gains_pending_notify_columns(tmp_path: Path) -> None:
    from assistai.jobs import Job

    path = tmp_path / "assistai.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE messages (
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
        CREATE TABLE allowlist (
            number TEXT PRIMARY KEY NOT NULL,
            admitted_at REAL NOT NULL
        );
        CREATE TABLE proposals (
            agent TEXT PRIMARY KEY NOT NULL,
            calls TEXT NOT NULL,
            tainted INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        );
        CREATE TABLE jobs (
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
            cancelled_at REAL
        );
        """
    )
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()

    store = Store(path)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="morning email",
            prompt="check",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=10.0,
            last_run_at=None,
            cancelled_at=None,
            pending_text="[job: morning email]\nheld",
            pending_after="advance",
        )
    )
    loaded = store.find_job("jacob", name="morning email")
    assert loaded is not None
    assert loaded.pending_text is not None
    assert "held" in loaded.pending_text
    assert loaded.once is False
    assert loaded.verbatim is False
    assert loaded.relay_sent is False
    store.close()


def test_schema_v11_gains_relay_sent(tmp_path: Path) -> None:
    """The Pi is already on the relay column. The sent flag has to be added."""
    from assistai.jobs import Job

    path = tmp_path / "assistai.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE jobs (
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
            relay INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute("PRAGMA user_version = 11")
    conn.commit()
    conn.close()

    store = Store(path)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="message to spouse",
            prompt="pick up milk",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=10.0,
            last_run_at=None,
            cancelled_at=None,
            relay=True,
        )
    )
    loaded = store.find_job("jacob", name="message to spouse")
    assert loaded is not None
    assert loaded.relay is True
    assert loaded.relay_sent is False
    assert loaded.relay_recorded is False
    assert store.mark_relay_sent(loaded.id) is True
    marked = store.find_job("jacob", name="message to spouse")
    assert marked is not None
    assert marked.relay_sent is True
    store.close()


def test_schema_v12_gains_relay_recorded(tmp_path: Path) -> None:
    """A phone that already has the message still needs the history record."""
    from assistai.jobs import Job

    path = tmp_path / "assistai.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE jobs (
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
            relay_sent INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute("PRAGMA user_version = 12")
    conn.commit()
    conn.close()

    store = Store(path)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="message to spouse",
            prompt="pick up milk",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=10.0,
            last_run_at=None,
            cancelled_at=None,
            relay=True,
            relay_sent=True,
        )
    )
    loaded = store.find_job("jacob", name="message to spouse")
    assert loaded is not None
    assert loaded.relay_sent is True
    assert loaded.relay_recorded is False
    store.close()


_V12_JOB_COLUMNS = """
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
    relay_sent INTEGER NOT NULL DEFAULT 0
"""


def _old_jobs_db(path: Path, *, version: int, columns: str) -> None:
    """A database at ``version`` holding one morning-email row written by that build."""
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE jobs ({columns})")
    conn.execute(
        "INSERT INTO jobs (id, agent, kind, name, prompt, every_seconds, created_at, "
        "next_run_at) VALUES ('jabcd1234', 'jacob', 'schedule', 'morning email', "
        "'summarize unread mail', 86400, 1.0, 10.0)"
    )
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    conn.close()


def test_schema_v12_keeps_an_existing_job_and_defaults_it_off(tmp_path: Path) -> None:
    """A wrong default would turn last month's digest into a relay or a raw text."""
    path = tmp_path / "assistai.sqlite"
    _old_jobs_db(path, version=12, columns=_V12_JOB_COLUMNS)

    store = Store(path)
    loaded = store.find_job("jacob", name="morning email")

    assert loaded is not None
    assert loaded.prompt == "summarize unread mail"
    assert loaded.next_run_at == 10.0
    assert not loaded.once
    assert not loaded.verbatim
    assert not loaded.relay
    assert not loaded.relay_sent
    assert not loaded.relay_recorded
    assert not loaded.tainted
    store.close()


def test_schema_v13_gains_job_taint_and_keeps_rows(tmp_path: Path) -> None:
    """Rows from before decision 0001 were treated as trusted, and still are."""
    path = tmp_path / "assistai.sqlite"
    columns = _V12_JOB_COLUMNS + ",\n    relay_recorded INTEGER NOT NULL DEFAULT 0\n"
    _old_jobs_db(path, version=13, columns=columns)

    store = Store(path)
    loaded = store.find_job("jacob", name="morning email")

    assert loaded is not None
    assert loaded.prompt == "summarize unread mail"
    assert not loaded.tainted
    # Reading falls back when the column is missing; a save does not.
    store.save_job(replace(loaded, id="jnew00001", name="tainted one", tainted=True))
    saved = store.find_job("jacob", name="tainted one")
    assert saved is not None
    assert saved.tainted
    store.close()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 15
    conn.close()


def test_active_job_names_are_unique(tmp_path: Path) -> None:
    from assistai.jobs import Job

    store = _store(tmp_path)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="morning email",
            prompt="check",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=10.0,
            last_run_at=None,
            cancelled_at=None,
        )
    )
    with pytest.raises(StoreError):
        store.save_job(
            Job(
                id="jffff9999",
                agent="jacob",
                kind="schedule",
                name="Morning Email",
                prompt="other",
                every_seconds=86400,
                ttl_seconds=None,
                created_at=2.0,
                expires_at=None,
                next_run_at=20.0,
                last_run_at=None,
                cancelled_at=None,
            )
        )
    store.cancel_job("jabcd1234", at=3.0)
    store.save_job(
        Job(
            id="jffff9999",
            agent="jacob",
            kind="schedule",
            name="morning email",
            prompt="again",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=4.0,
            expires_at=None,
            next_run_at=30.0,
            last_run_at=None,
            cancelled_at=None,
        )
    )
    found = store.find_job("jacob", name="morning email")
    assert found is not None
    assert found.id == "jffff9999"
    store.close()


def _proposal(
    agent: str,
    arguments: str,
    *,
    created_at: float = 1.0,
    expires_at: float = 2.0,
    tainted: bool = False,
) -> Proposal:
    return Proposal(
        agent=agent,
        calls=(ToolCall(id="c1", name="shared_write", arguments=arguments),),
        tainted=tainted,
        created_at=created_at,
        expires_at=expires_at,
    )


def test_a_backward_clock_step_expires_a_proposal(tmp_path: Path) -> None:
    """A clock that moved back before the proposal was made cannot be trusted to time it out."""
    store = _store(tmp_path)
    made = 1_800_000_000.0
    store.replace_proposal(_proposal("jacob", "{}", created_at=made, expires_at=made + 600))

    live, expired = store.take_proposal("jacob", now=made - 3600)

    assert live is None
    assert expired is True
    store.close()


def test_a_proposal_made_up_to_a_minute_ahead_is_still_live(tmp_path: Path) -> None:
    """NTP slews of a few seconds must not throw away a proposal the person is about to confirm."""
    store = _store(tmp_path)
    made = 1_800_000_000.0
    store.replace_proposal(_proposal("jacob", "{}", created_at=made, expires_at=made + 600))

    live, expired = store.take_proposal("jacob", now=made - 60)

    assert live is not None
    assert expired is False
    store.close()


def test_the_newest_event_is_none_on_a_new_install(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.newest_event_at() is None
    store.close()


def _event_job(*, created_at: float, last_run_at: float | None, next_run_at: float) -> Job:
    return Job(
        id="jabcd1234",
        agent="jacob",
        kind="schedule",
        name="digest",
        prompt="summarize",
        every_seconds=86400,
        ttl_seconds=None,
        created_at=created_at,
        expires_at=None,
        next_run_at=next_run_at,
        last_run_at=last_run_at,
        cancelled_at=None,
    )


@pytest.mark.parametrize("newest", ["message", "proposal", "job_created", "job_run"])
def test_the_newest_event_reads_every_past_event_column(tmp_path: Path, newest: str) -> None:
    """Any one of the four can be the latest thing the household did before a restart."""
    store = _store(tmp_path)
    at = {name: 100.0 for name in ("message", "proposal", "job_created", "job_run")}
    at[newest] = 500.0
    store.save_history("jacob", [Message(role="user", content="hi", created_at=at["message"])])
    store.replace_proposal(
        _proposal("jacob", "{}", created_at=at["proposal"], expires_at=9_000_000_000)
    )
    store.save_job(
        _event_job(
            created_at=at["job_created"], last_run_at=at["job_run"], next_run_at=8_000_000_000
        )
    )

    assert store.newest_event_at() == 500.0
    store.close()


def test_the_newest_event_ignores_future_schedule_columns_and_unrun_jobs(tmp_path: Path) -> None:
    """next_run_at and expires_at are ahead by design and must not look like a fast clock."""
    store = _store(tmp_path)
    store.save_job(_event_job(created_at=200.0, last_run_at=None, next_run_at=8_000_000_000))

    assert store.newest_event_at() == 200.0
    store.close()


# The schema 0.1.8 writes, copied from bbd205e. A real file shaped like the
# Pi's is the only fair test of the migration and of the downgrade back.
_V14_SCHEMA = """
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
    tainted INTEGER NOT NULL DEFAULT 0
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

# 0.1.8's save_job column list, exactly. A row written after a downgrade has
# no schedule columns.
_V14_JOB_INSERT = (
    "INSERT INTO jobs (id, agent, kind, name, prompt, every_seconds, "
    "ttl_seconds, created_at, expires_at, next_run_at, last_run_at, "
    "cancelled_at, pending_text, pending_untrusted, pending_after, "
    "pending_delivered, once, verbatim, relay, relay_sent, relay_recorded, "
    "tainted) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_MGMPLUS_PROMPT = "Cancel MGM+ before the rate goes up."
_MGMPLUS_FIRST = chicago(2026, 12, 10, 8)


def _v14_row(
    job_id: str,
    name: str,
    *,
    every_seconds: int = 86400,
    next_run_at: float,
    last_run_at: float | None = None,
    cancelled_at: float | None = None,
    prompt: str = "summarize unread mail",
    kind: str = "schedule",
    ttl_seconds: int | None = None,
    expires_at: float | None = None,
    pending_text: str | None = None,
    pending_after: str | None = None,
    pending_delivered: int = 0,
    once: int = 0,
    verbatim: int = 0,
    relay: int = 0,
) -> tuple[object, ...]:
    """One row in 0.1.8's column order."""
    return (
        job_id,
        "jacob",
        kind,
        name,
        prompt,
        every_seconds,
        ttl_seconds,
        chicago(2026, 10, 18, 20, 0),
        expires_at,
        next_run_at,
        last_run_at,
        cancelled_at,
        pending_text,
        0,
        pending_after,
        pending_delivered,
        once,
        verbatim,
        relay,
        0,
        0,
        0,
    )


def _mgmplus_row() -> tuple[object, ...]:
    """The one live job on the Pi: daily, never run, first fire Dec 10 at 8:00 AM CST."""
    return _v14_row(
        "j00000001",
        "mgmplus cancel reminder",
        next_run_at=_MGMPLUS_FIRST,
        prompt=_MGMPLUS_PROMPT,
        verbatim=1,
    )


def _v14_db(path: Path, *rows: tuple[object, ...]) -> None:
    """A version 14 file, as 0.1.8 leaves it, holding ``rows``."""
    conn = sqlite3.connect(path)
    conn.executescript(_V14_SCHEMA)
    conn.executemany(_V14_JOB_INSERT, rows)
    conn.execute("PRAGMA user_version = 14")
    conn.commit()
    conn.close()


def _job_row(path: Path, job_id: str) -> sqlite3.Row:
    """A row read without the store, so cancelled rows and raw columns are visible."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    assert row is not None
    return cast(sqlite3.Row, row)


def _user_version(path: Path) -> int:
    conn = sqlite3.connect(path)
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    conn.close()
    return version


def test_schema_v14_gains_wall_clock_schedules_and_keeps_rows(tmp_path: Path) -> None:
    """The Pi's file upgrades with every row, and the mgmplus reminder still fires Dec 10 at 8."""
    path = tmp_path / "assistai.sqlite"
    email_ran = chicago(2026, 10, 30, 21, 12)
    watch_ran = chicago(2026, 11, 8, 9, 0)
    relay_at = chicago(2026, 11, 9, 18, 30)
    _v14_db(
        path,
        _mgmplus_row(),
        _v14_row(
            "j00000002",
            "morning email",
            next_run_at=email_ran + 86400,
            last_run_at=email_ran,
            cancelled_at=chicago(2026, 10, 30, 21, 40),
        ),
        _v14_row(
            "j00000003",
            "pool reminder",
            next_run_at=chicago(2026, 10, 25, 9),
            cancelled_at=chicago(2026, 10, 25, 9, 1),
            prompt="Put the cleaner in the pool.",
            pending_text="[job: pool reminder]\nPut the cleaner in the pool.",
            pending_after="cancel",
            pending_delivered=1,
            once=1,
            verbatim=1,
        ),
        _v14_row(
            "j00000004",
            "flight watch",
            kind="watch",
            every_seconds=60,
            ttl_seconds=86400,
            expires_at=watch_ran + 86400,
            next_run_at=watch_ran + 60,
            last_run_at=watch_ran,
            prompt="Denver fares under $200",
        ),
        _v14_row(
            "j00000005",
            "message to spouse",
            next_run_at=relay_at,
            prompt="pick up milk",
            once=1,
            verbatim=1,
            relay=1,
        ),
    )

    store = Store(path)
    with capture_logs() as logs:
        filled = store.backfill_job_schedules("America/Chicago", now=chicago(2026, 11, 9, 12))
    store.close()

    assert filled == 5
    assert _user_version(path) == SCHEMA_VERSION == 15
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE zone IS NULL").fetchone()[0] == 0
    conn.close()

    mgmplus = _job_row(path, "j00000001")
    assert mgmplus["every_days"] == 1
    assert mgmplus["at_local"] == "08:00"
    assert mgmplus["every_seconds"] == 86400
    assert mgmplus["next_run_at"] == _MGMPLUS_FIRST
    assert wall(mgmplus["next_run_at"]) == "2026-12-10 08:00:00 CST"
    assert mgmplus["anchor_at"] == _MGMPLUS_FIRST
    assert mgmplus["cancelled_at"] is None

    email = _job_row(path, "j00000002")
    assert email["cancelled_at"] == chicago(2026, 10, 30, 21, 40)
    assert email["zone"] == "America/Chicago"
    assert email["every_days"] == 1
    assert email["at_local"] == "21:12"

    pool = _job_row(path, "j00000003")
    assert pool["cancelled_at"] == chicago(2026, 10, 25, 9, 1)
    assert pool["zone"] == "America/Chicago"
    assert pool["pending_text"] == "[job: pool reminder]\nPut the cleaner in the pool."
    assert pool["pending_after"] == "cancel"
    assert pool["pending_delivered"] == 1
    assert pool["every_days"] is None

    watch = _job_row(path, "j00000004")
    assert watch["every_days"] is None
    assert watch["at_local"] is None
    assert watch["every_seconds"] == 60
    assert watch["next_run_at"] == watch_ran + 60
    assert watch["anchor_at"] == watch_ran

    relay = _job_row(path, "j00000005")
    assert relay["zone"] == "America/Chicago"
    assert relay["every_days"] is None
    assert relay["at_local"] is None
    assert relay["next_run_at"] == relay_at
    assert relay["anchor_at"] == relay_at

    migrated = [entry for entry in logs if entry["event"] == "store.job_schedule_migrated"]
    assert sorted(entry["job"] for entry in migrated) == [f"j0000000{n}" for n in range(1, 6)]
    by_job = {entry["job"]: entry for entry in migrated}
    assert by_job["j00000001"]["every_days"] == 1
    assert by_job["j00000001"]["at_local"] == "08:00"
    assert by_job["j00000001"]["cancelled"] is False
    assert by_job["j00000002"]["cancelled"] is True
    for prompt in (_MGMPLUS_PROMPT, "pick up milk", "Put the cleaner", "Denver"):
        assert prompt not in repr(migrated)


@pytest.mark.parametrize(
    "now",
    [
        pytest.param(_MGMPLUS_FIRST - 86400, id="a day before"),
        pytest.param(_MGMPLUS_FIRST - 60, id="a minute before"),
        pytest.param(_MGMPLUS_FIRST + 3600, id="an hour after"),
    ],
)
def test_a_legacy_row_that_never_ran_keeps_its_first_fire(tmp_path: Path, now: float) -> None:
    """Nothing stepped it, so the stored first fire is the slot the person chose.

    An hour late it is simply due and runs at the next tick; it is never
    moved to the following day.
    """
    path = tmp_path / "assistai.sqlite"
    _v14_db(path, _mgmplus_row())
    store = Store(path)

    assert store.backfill_job_schedules("America/Chicago", now=now) == 1
    loaded = store.find_job("jacob", name="mgmplus cancel reminder")
    store.close()

    assert loaded is not None
    assert loaded.next_run_at == _MGMPLUS_FIRST
    assert loaded.anchor_at == _MGMPLUS_FIRST
    assert loaded.every_days == 1
    assert loaded.at_local == "08:00"


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        pytest.param(chicago(2026, 10, 31, 20), "2026-11-01 08:00:00 CST", id="the evening before"),
        pytest.param(chicago(2026, 11, 1, 6, 30), "2026-11-01 08:00:00 CST", id="before 7"),
        pytest.param(chicago(2026, 11, 1, 7, 30), "2026-11-01 08:00:00 CST", id="between 7 and 8"),
        pytest.param(chicago(2026, 11, 9, 12), "2026-11-10 08:00:00 CST", id="days later"),
    ],
)
def test_a_legacy_daily_row_that_ran_keeps_its_hour(
    tmp_path: Path, now: float, expected: str
) -> None:
    """0.1.8 stepped 8:00 AM CDT by 86,400 seconds to 7:00 AM CST on Nov 1.

    The slot before the stored one is the hour the person chose. The drifted
    7:00 is dropped, so the row texts once that morning, at 8:00.
    """
    path = tmp_path / "assistai.sqlite"
    ran = chicago(2026, 10, 31, 8)
    _v14_db(
        path,
        _v14_row("j00000001", "morning email", next_run_at=ran + 86400, last_run_at=ran),
    )
    assert wall(ran + 86400) == "2026-11-01 07:00:00 CST"
    store = Store(path)

    store.backfill_job_schedules("America/Chicago", now=now)
    loaded = store.find_job("jacob", name="morning email")
    store.close()

    assert loaded is not None
    assert loaded.at_local == "08:00"
    assert loaded.every_days == 1
    assert loaded.anchor_at == ran
    assert wall(loaded.next_run_at) == expected
    assert loaded.next_run_at > now


def test_an_elapsed_legacy_row_keeps_its_next_run_and_backfill_runs_once(tmp_path: Path) -> None:
    """A 90-minute gap is not tied to the clock face, and a second start changes nothing."""
    path = tmp_path / "assistai.sqlite"
    ran = chicago(2026, 11, 9, 9, 0)
    drifted = ran + 5400 + 17
    _v14_db(
        path,
        _v14_row(
            "j00000001", "price check", every_seconds=5400, next_run_at=drifted, last_run_at=ran
        ),
    )
    store = Store(path)

    assert store.backfill_job_schedules("America/Chicago", now=chicago(2026, 11, 9, 10)) == 1
    first = store.find_job("jacob", name="price check")
    with capture_logs() as logs:
        again = store.backfill_job_schedules("America/Chicago", now=chicago(2026, 11, 9, 11))
    second = store.find_job("jacob", name="price check")
    store.close()

    assert first is not None
    assert first.next_run_at == drifted
    assert first.every_seconds == 5400
    assert first.every_days is None
    assert first.at_local is None
    assert first.anchor_at == drifted - 5400
    assert again == 0
    assert second == first
    assert not any(entry["event"] == "store.job_schedule_migrated" for entry in logs)


def test_a_new_job_round_trips_its_schedule(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="bins",
            prompt="Take the bins in.",
            every_seconds=7 * 86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=chicago(2026, 12, 7, 19),
            last_run_at=None,
            cancelled_at=None,
            zone="America/Chicago",
            anchor_at=chicago(2026, 12, 7, 19),
            every_days=7,
            at_local="19:00",
        )
    )

    loaded = store.find_job("jacob", name="bins")
    store.close()

    assert loaded is not None
    assert loaded.schedule() == Schedule(
        zone=CHICAGO, anchor_at=chicago(2026, 12, 7, 19), every_days=7, at_local="19:00"
    )


def test_reschedule_writes_the_schedule_with_the_next_run(tmp_path: Path) -> None:
    """A whole-day row moved to minutes must not keep repeating on its old clock time."""
    store = _store(tmp_path)
    start = chicago(2026, 12, 7, 8)
    store.save_job(
        Job(
            id="jabcd1234",
            agent="jacob",
            kind="schedule",
            name="bins",
            prompt="Take the bins in.",
            every_seconds=86400,
            ttl_seconds=None,
            created_at=1.0,
            expires_at=None,
            next_run_at=start,
            last_run_at=None,
            cancelled_at=None,
            zone="America/Chicago",
            anchor_at=start,
            every_days=1,
            at_local="08:00",
        )
    )

    assert store.reschedule_job(
        "jabcd1234",
        every_seconds=5400,
        next_run_at=start + 5400,
        schedule=Schedule(zone=CHICAGO, anchor_at=start + 5400, every_seconds=5400),
    )
    loaded = store.find_job("jacob", name="bins")
    store.close()

    assert loaded is not None
    assert loaded.every_seconds == 5400
    assert loaded.next_run_at == start + 5400
    assert loaded.anchor_at == start + 5400
    assert loaded.every_days is None
    assert loaded.at_local is None
    assert loaded.zone == "America/Chicago"


def test_downgrade_lets_a_v14_writer_add_a_job_and_forward_again_backfills_it(
    tmp_path: Path,
) -> None:
    """Rollback is the previous image. A job it creates is filled in on the way back."""
    path = tmp_path / "assistai.sqlite"
    _v14_db(path, _mgmplus_row())
    store = Store(path)
    store.backfill_job_schedules("America/Chicago", now=chicago(2026, 11, 9, 12))
    store.close()

    downgrade(path, to=14)

    assert _user_version(path) == 14
    # 0.1.8 opens a version 14 file without touching it, then writes its own columns.
    conn = sqlite3.connect(path)
    conn.execute(
        _V14_JOB_INSERT,
        _v14_row(
            "j00000009",
            "bins",
            every_seconds=7 * 86400,
            next_run_at=chicago(2026, 11, 12, 19),
            prompt="Take the bins in.",
            verbatim=1,
        ),
    )
    conn.commit()
    conn.close()

    store = Store(path)
    assert _user_version(path) == 15
    with capture_logs() as logs:
        filled = store.backfill_job_schedules("America/Chicago", now=chicago(2026, 11, 10, 12))
    bins = store.find_job("jacob", name="bins")
    mgmplus = store.find_job("jacob", name="mgmplus cancel reminder")
    store.close()

    assert filled == 1
    assert [entry["job"] for entry in logs if entry["event"] == "store.job_schedule_migrated"] == [
        "j00000009"
    ]
    assert bins is not None
    assert bins.zone == "America/Chicago"
    assert bins.every_days == 7
    assert bins.at_local == "19:00"
    assert bins.next_run_at == chicago(2026, 11, 12, 19)
    assert mgmplus is not None
    assert mgmplus.next_run_at == _MGMPLUS_FIRST
    assert mgmplus.at_local == "08:00"


def test_a_fresh_file_can_be_downgraded(tmp_path: Path) -> None:
    """A fresh file and a migrated one differ in their CREATE text, not in their shape."""
    path = tmp_path / "assistai.sqlite"
    _store(tmp_path).close()

    downgrade(path, to=14)

    assert _user_version(path) == 14


def _extra_column(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE jobs ADD COLUMN surprise TEXT")


def _extra_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE surprise (id INTEGER PRIMARY KEY)")


def _extra_index(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX jobs_kind ON jobs(kind)")


def _missing_column(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE jobs DROP COLUMN at_local")


def _retyped_column(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE jobs DROP COLUMN at_local")
    conn.execute("ALTER TABLE jobs ADD COLUMN at_local INTEGER")


def _version(number: int) -> Callable[[sqlite3.Connection], None]:
    def change(conn: sqlite3.Connection) -> None:
        conn.execute(f"PRAGMA user_version = {number}")

    return change


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_extra_column, id="extra column"),
        pytest.param(_extra_table, id="extra table"),
        pytest.param(_extra_index, id="extra index"),
        pytest.param(_missing_column, id="missing column"),
        pytest.param(_retyped_column, id="retyped column"),
        pytest.param(_version(14), id="already 14"),
        pytest.param(_version(16), id="newer"),
    ],
)
def test_downgrade_refuses_a_file_it_does_not_recognise(
    tmp_path: Path, change: Callable[[sqlite3.Connection], None]
) -> None:
    """An older build could misread a change it never made, so the file is left as it is."""
    path = tmp_path / "assistai.sqlite"
    _store(tmp_path).close()
    conn = sqlite3.connect(path)
    change(conn)
    conn.commit()
    before = int(conn.execute("PRAGMA user_version").fetchone()[0])
    conn.close()

    with pytest.raises(StoreError, match="refusing to downgrade"):
        downgrade(path, to=14)

    assert _user_version(path) == before


def test_downgrade_refuses_an_unknown_target_and_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "assistai.sqlite"

    with pytest.raises(StoreError, match="no state database"):
        downgrade(path, to=14)
    assert not path.exists()

    _store(tmp_path).close()
    with pytest.raises(StoreError, match="cannot downgrade to version 13"):
        downgrade(path, to=13)
    assert _user_version(path) == 15
