from __future__ import annotations

from assistai.inference.types import ToolCall
from assistai.staging import format_executed, format_proposal, parse_decision


def test_only_a_whole_message_yes_confirms() -> None:
    assert parse_decision("yes") == "confirm"
    assert parse_decision("YES") == "confirm"
    assert parse_decision("confirm") == "confirm"
    assert parse_decision("no") == "reject"
    assert parse_decision("cancel") == "reject"
    assert parse_decision("y") is None
    assert parse_decision("ok") is None
    assert parse_decision("okay") is None
    assert parse_decision("yes please") is None
    assert parse_decision("please yes") is None
    assert parse_decision("/approve 123456") is None


def test_preview_is_formatted_from_stored_calls_not_prose() -> None:
    preview = format_proposal(
        (
            ToolCall(id="c1", name="shared_write", arguments='{"path": "list"}'),
            ToolCall(id="c2", name="shared_write", arguments='{"path": "notes"}'),
        ),
        tainted=True,
    )

    assert "shared_write" in preview
    assert '"path": "list"' in preview
    assert '"path": "notes"' in preview
    assert "untrusted" in preview
    assert "yes" in preview.lower()


def test_executed_summary_keeps_call_order() -> None:
    summary = format_executed(
        [
            (ToolCall(id="c1", name="shared_write", arguments="{}"), '{"ok": true}'),
            (ToolCall(id="c2", name="shared_write", arguments="{}"), '{"ok": false}'),
        ]
    )

    assert summary.startswith("Ran 2 actions:")
    assert summary.index('{"ok": true}') < summary.index('{"ok": false}')
