from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistai.broker import Scope, ToolBroker, builtin_catalog
from assistai.inference.types import ToolCall
from assistai.jobs import bind_jobs
from assistai.staging import format_executed, format_proposal, parse_decision
from assistai.store import Store
from tests.agent_fakes import agent, household
from tests.clock_fakes import FakeClock, chicago, chicago_clock, wall
from tests.fakes import body


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
                    '"prompt": "summarize important mail", "every_days": 1, '
                    '"at": {"date": "2026-10-31", "hour": 8, "minute": 0}}'
                ),
            ),
        ),
        tainted=False,
    )

    assert "morning email" in preview
    assert "summarize important mail" in preview
    assert "First run: Saturday, October 31, 2026 at 8:00 AM" in preview
    assert "Repeats: every day at 8:00 AM" in preview
    assert "yes" in preview.lower()


def test_a_delayed_job_preview_counts_from_the_yes() -> None:
    preview = format_proposal(
        (
            ToolCall(
                id="c1",
                name="job_create",
                arguments=(
                    '{"kind": "schedule", "name": "inbox sweep", '
                    '"prompt": "summarize important mail", "every_minutes": 90, '
                    '"delay": {"hours": 2}}'
                ),
            ),
        ),
        tainted=False,
    )

    assert "First run: 2 hours after you reply yes" in preview
    assert "Repeats: every 90 minutes" in preview


async def test_a_relative_day_is_fixed_when_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'Tomorrow' at 11:58 PM is Saturday. A yes at 12:03 AM still runs Saturday, as previewed."""
    clock = FakeClock(chicago(2026, 10, 30, 23, 58))
    monkeypatch.setattr("assistai.jobs._now", clock)
    store = Store(tmp_path / "assistai.sqlite")
    catalog = builtin_catalog()
    for name, handler in bind_jobs(store, timezone="America/Chicago").items():
        catalog.bind(name, handler)
    jacob = agent("jacob", "+15555550101", tools=("job_create", "get_time"))
    broker = ToolBroker(
        catalog,
        household(jacob, agent("spouse", "+15555550102")).broker,
        store=store,
        clock=chicago_clock(clock),
    )
    surface = broker.for_agent(jacob, Scope(tainted=True))
    await body(surface, ToolCall(id="c0", name="get_time", arguments="{}"))
    await body(
        surface,
        ToolCall(
            id="c1",
            name="job_create",
            arguments=json.dumps(
                {
                    "kind": "schedule",
                    "name": "bins",
                    "prompt": "Take the bins in.",
                    "once": True,
                    "verbatim": True,
                    "at": {"day": "tomorrow", "hour": 8},
                }
            ),
        ),
    )
    staged = surface.take_staged()
    preview = format_proposal(staged, tainted=True)

    clock.set(chicago(2026, 10, 31, 0, 3))
    yes = broker.for_agent(jacob, Scope(tainted=True))
    for call in staged:
        await yes.commit(call)
    saved = store.find_job("jacob", name="bins")
    store.close()

    assert "First run: Saturday, October 31, 2026 at 8:00 AM" in preview
    assert saved is not None
    assert wall(saved.next_run_at) == "2026-10-31 08:00:00 CDT"


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
                    '"prompt": "summarize important mail", "every_minutes": 60}'
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
