from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistai.broker import Scope, ToolBroker, builtin_catalog
from assistai.inference.types import ToolCall
from assistai.jobs import Job, bind_jobs
from assistai.staging import format_executed, format_proposal, parse_decision
from assistai.store import Store
from tests.agent_fakes import agent, household
from tests.clock_fakes import FakeClock, chicago, chicago_clock, wall
from tests.fakes import body

# Friday, October 30, 2026 at 3:04 PM in Chicago: the afternoon the audit's
# requests were made.
FRIDAY = chicago_clock(chicago(2026, 10, 30, 15, 4))


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
        clock=FRIDAY,
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
        clock=FRIDAY,
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
        clock=FRIDAY,
    )

    assert "morning email" in preview
    assert "summarize important mail" in preview
    assert "First run: tomorrow at 8:00 AM (Saturday, October 31)" in preview
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
        clock=FRIDAY,
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
    preview = format_proposal(staged, tainted=True, clock=chicago_clock(clock))

    clock.set(chicago(2026, 10, 31, 0, 3))
    yes = broker.for_agent(jacob, Scope(tainted=True))
    for call in staged:
        await yes.commit(call)
    saved = store.find_job("jacob", name="bins")
    store.close()

    assert "First run: tomorrow at 8:00 AM (Saturday, October 31)" in preview
    assert saved is not None
    assert wall(saved.next_run_at) == "2026-10-31 08:00:00 CDT"


def test_a_calendar_add_preview_shows_the_stored_event() -> None:
    preview = format_proposal(
        (
            ToolCall(
                id="c1",
                name="calendar_add",
                arguments=(
                    '{"summary": "Dentist", "start": "2026-11-05T15:00", "location": "Office"}'
                ),
            ),
        ),
        tainted=False,
        clock=FRIDAY,
    )

    assert "Dentist" in preview
    assert "Thursday at 3:00 PM (November 5) to 4:00 PM" in preview
    assert "Office" in preview
    assert "yes" in preview.lower()


def test_a_job_cancel_preview_names_the_job() -> None:
    preview = format_proposal(
        (ToolCall(id="c1", name="job_cancel", arguments='{"name": "morning email"}'),),
        tainted=False,
        clock=FRIDAY,
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
        clock=FRIDAY,
    )

    assert "morning email" in preview
    assert "cancel" in preview.lower()
    assert "summarize important mail" in preview
    assert "`job_cancel`" not in preview
    assert "yes" in preview.lower()


def test_every_job_preview_names_the_first_run() -> None:
    """Single or bundled, a job or relay preview says when it first runs (B-2, D-9, X3-6, B-6)."""
    bins = _call(
        "job_create",
        kind="schedule",
        name="bins",
        prompt="Take the bins in.",
        once=True,
        verbatim=True,
        at={"date": "2026-10-31", "hour": 8},
    )
    sweep = _call(
        "job_create",
        kind="schedule",
        name="inbox sweep",
        prompt="summarize important mail",
        every_minutes=90,
    )
    milk = _call("relay", body="pick up milk", at={"date": "2026-10-30", "hour": 20})
    soon = _call("relay", body="leaving now", delay={"minutes": 2})

    def preview(*calls: ToolCall) -> str:
        return format_proposal(calls, tainted=False, clock=FRIDAY)

    assert "First run: tomorrow at 8:00 AM (Saturday, October 31)" in preview(bins)
    assert "First run: 90 minutes after you reply yes" in preview(sweep)
    assert "to the other phone today at 8:00 PM when you confirm" in preview(milk)
    assert "to the other phone 2 minutes after you reply yes:" in preview(soon)

    bundle = preview(bins, sweep, milk, soon)
    assert "first run tomorrow at 8:00 AM (Saturday, October 31), repeats once" in bundle
    assert "first run 90 minutes after you reply yes, repeats every 90 minutes" in bundle
    assert "Send to the other phone today at 8:00 PM: pick up milk" in bundle
    assert "Send to the other phone 2 minutes after you reply yes: leaving now" in bundle
    assert "`relay`" not in bundle


def test_a_reschedule_preview_names_the_next_run() -> None:
    """Moving the December 10 reminder names the run the yes keeps, not just the new gap."""
    mgmplus = Job(
        id="jmgmplus1",
        agent="jacob",
        kind="schedule",
        name="mgmplus cancel reminder",
        prompt="Cancel MGM+ before the rate goes up.",
        every_seconds=86400,
        ttl_seconds=None,
        created_at=chicago(2026, 10, 30, 21, 12),
        expires_at=None,
        next_run_at=chicago(2026, 12, 10, 8),
        last_run_at=None,
        cancelled_at=None,
        verbatim=True,
        zone="America/Chicago",
        anchor_at=chicago(2026, 12, 10, 8),
        every_days=1,
        at_local="08:00",
    )

    def jobs(*, name: str | None, job_id: str | None) -> Job | None:
        return mgmplus if name == mgmplus.name else None

    weekly = _call("job_reschedule", name=mgmplus.name, every_days=7)
    at_nine = _call("job_reschedule", name=mgmplus.name, at={"hour": 9})

    single = format_proposal((weekly,), tainted=False, clock=FRIDAY, jobs=jobs)
    bundle = format_proposal((weekly, at_nine), tainted=False, clock=FRIDAY, jobs=jobs)

    assert "Next run: Thursday, December 10 at 8:00 AM, then every Thursday at 8:00 AM" in single
    assert (
        "Change mgmplus cancel reminder: "
        "Next run: Thursday, December 10 at 9:00 AM, then every day at 9:00 AM"
    ) in bundle


def test_a_digest_preview_names_its_first_date() -> None:
    """A named start date is in the preview, and one the call left out is visible (C-7)."""
    dated = _call(
        "job_create",
        kind="schedule",
        name="denver flights",
        prompt="Starting December 9, check flight prices to Denver.",
        every_days=1,
        at={"date": "2026-12-09", "hour": 7},
    )
    undated = _call(
        "job_create",
        kind="schedule",
        name="denver flights",
        prompt="Starting December 9, check flight prices to Denver.",
        every_days=1,
        at={"date": "2026-10-31", "hour": 7},
    )

    assert "First run: Wednesday, December 9 at 7:00 AM" in format_proposal(
        (dated,), tainted=False, clock=FRIDAY
    )
    # The prompt says December 9; the preview says what the yes runs.
    assert "First run: tomorrow at 7:00 AM (Saturday, October 31)" in format_proposal(
        (undated,), tainted=False, clock=FRIDAY
    )


def _call(tool: str, **arguments: object) -> ToolCall:
    return ToolCall(id=f"c-{tool}", name=tool, arguments=json.dumps(arguments))
