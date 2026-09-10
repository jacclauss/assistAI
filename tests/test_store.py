from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from assistai.errors import StoreError
from assistai.inference.types import Message, ToolCall
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
