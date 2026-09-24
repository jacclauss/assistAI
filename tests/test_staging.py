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


def test_a_relay_preview_shows_the_stored_body() -> None:
    preview = format_proposal(
        (ToolCall(id="c1", name="relay", arguments='{"body": "pick up milk"}'),),
        tainted=False,
    )

    assert "pick up milk" in preview
    assert "Queued" not in preview
    assert "yes" in preview.lower()
    assert "what to change" in preview.lower()
    assert "relay from you" in preview.lower()


def test_a_job_create_preview_shows_the_stored_schedule() -> None:
    preview = format_proposal(
        (
            ToolCall(
                id="c1",
                name="job_create",
                arguments=(
                    '{"kind": "schedule", "name": "morning email", '
                    '"prompt": "summarize important mail", "every_seconds": 86400}'
                ),
            ),
        ),
        tainted=False,
    )

    assert "morning email" in preview
    assert "summarize important mail" in preview
    assert "86400" in preview
    assert "yes" in preview.lower()


def test_a_calendar_add_preview_shows_the_stored_event() -> None:
    preview = format_proposal(
        (
            ToolCall(
                id="c1",
                name="calendar_add",
                arguments=(
                    '{"summary": "Dentist", "start": "2026-09-24T15:00", "location": "Office"}'
                ),
            ),
        ),
        tainted=False,
    )

    assert "Dentist" in preview
    assert "2026-09-24 15:00-16:00" in preview
    assert "Office" in preview
    assert "yes" in preview.lower()


def test_a_job_cancel_preview_names_the_job() -> None:
    preview = format_proposal(
        (ToolCall(id="c1", name="job_cancel", arguments='{"name": "morning email"}'),),
        tainted=False,
    )

    assert "morning email" in preview
    assert "cancel" in preview.lower()
    assert "yes" in preview.lower()


def test_cancel_then_create_preview_names_the_jobs() -> None:
    preview = format_proposal(
        (
            ToolCall(id="c1", name="job_cancel", arguments='{"name": "morning email"}'),
            ToolCall(
                id="c2",
                name="job_create",
                arguments=(
                    '{"kind": "schedule", "name": "morning email", '
                    '"prompt": "summarize important mail", "every_seconds": 86400}'
                ),
            ),
        ),
        tainted=False,
    )

    assert "morning email" in preview
    assert "cancel" in preview.lower()
    assert "summarize important mail" in preview
    assert "`job_cancel`" not in preview
    assert "yes" in preview.lower()
