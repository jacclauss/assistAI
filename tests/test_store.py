from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from assistai.errors import StoreError
from assistai.inference.types import Message, ToolCall
from assistai.staging import Proposal
from assistai.store import SCHEMA_VERSION, Store


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
    assert store.reschedule_job("jabcd1234", every_seconds=600, next_run_at=610.0)

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
    store.close()


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
