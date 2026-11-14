"""The eval harness without the model: case files parse and the grader is honest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from assistai.inference.tools import bind_clock
from evals.harness import CASES_DIR, Case, CaseError, grade, load_cases, run_case
from tests.clock_fakes import chicago
from tests.fakes import (
    Handler,
    client_for,
    completion_stream,
    pin,
    recorded,
    sequence,
    text_event,
    tool_event,
)

_TIME_CASES = (
    "reminder-named-clock",
    "reminder-tomorrow-late-night",
    "job-morning-email",
    "job-morning-email-named-hour",
    "job-dated-digest",
    "job-list-when",
    "job-reschedule-hour",
)


def _roster(repo_root: Path) -> Path:
    return repo_root / "config" / "assistai.example.toml"


def _case(repo_root: Path, case_id: str) -> Case:
    return next(
        case for case in load_cases(CASES_DIR, roster=_roster(repo_root)) if case.id == case_id
    )


def _write(tmp_path: Path, text: str) -> Path:
    (tmp_path / "bad.toml").write_text('id = "bad"\nuser = "hi"\n' + text, encoding="utf-8")
    return tmp_path


def _calls_then_text(*calls: tuple[str, dict[str, Any]], reply: str = "Done.") -> Handler:
    """A model that makes each call in its own round, then answers."""
    rounds = [
        completion_stream(
            tool_event(
                call_id=f"c{index}",
                name=name,
                arguments=json.dumps(arguments),
                finish="tool_calls",
            )
        )
        for index, (name, arguments) in enumerate(calls, start=1)
    ]
    return sequence(*rounds, completion_stream(text_event(reply, finish="stop")))


def _sent(request: httpx.Request) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = json.loads(request.content)["messages"]
    return messages


def test_every_case_file_loads(repo_root: Path) -> None:
    cases = load_cases(CASES_DIR, roster=_roster(repo_root))

    assert cases
    assert all("regression" in case.tags for case in cases)


def test_the_time_cases_run_on_a_chicago_clock(repo_root: Path) -> None:
    """Spec 016 names seven time cases, each on the household's zone and a fixed moment."""
    cases = {case.id: case for case in load_cases(CASES_DIR, roster=_roster(repo_root))}

    for case_id in _TIME_CASES:
        assert cases[case_id].timezone == "America/Chicago", case_id
        assert cases[case_id].now is not None, case_id
    assert cases["reminder-named-clock"].now == chicago(2026, 10, 30, 15, 4)
    assert cases["reminder-tomorrow-late-night"].now == chicago(2026, 10, 30, 23, 30)
    named = cases["reminder-named-clock"].expect.arguments["job_create"]
    assert dict(named.equals) == {"at.hour": 20, "at.minute": 53}
    assert named.absent == ("delay",)
    late = cases["reminder-tomorrow-late-night"].expect.arguments["job_create"]
    assert [dict(option) for option in late.any_of] == [
        {"at.day": "tomorrow"},
        {"at.date": "2026-10-31"},
    ]
    (mgmplus,) = cases["job-list-when"].jobs
    assert mgmplus.every_days == 1
    assert mgmplus.at_local == "08:00"
    assert mgmplus.next_run_at == chicago(2026, 12, 10, 8)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("now = 2026-10-30T15:04:00\n", "go together"),
        ('timezone = "America/Chicago"\n', "go together"),
        ('now = "3:04 PM"\ntimezone = "America/Chicago"\n', "local date-time"),
        ('now = 2026-10-30T15:04:00-05:00\ntimezone = "America/Chicago"\n', "local date-time"),
        ('now = 2027-03-14T02:30:00\ntimezone = "America/Chicago"\n', "now:"),
        ('now = 2026-10-30T15:04:00\ntimezone = "Mars/Olympus"\n', "unknown timezone"),
        ('[[jobs]]\nname = "x"\nprompt = "y"\nnext_run = 2026-12-10T08:00:00\n', "need now"),
        (
            'now = 2026-10-30T15:04:00\ntimezone = "America/Chicago"\n'
            '[fixtures]\nget_time = "{}"\n',
            "not a fixture",
        ),
        ("[expect.arguments.job_create]\n", "expects nothing"),
        ("[expect.arguments.job_create]\nhour = 8\n", "takes equals, absent, and any_of"),
        ('[expect.arguments.job_create]\nequals = { "at hour" = 8 }\n', "dotted path"),
        ("[expect.arguments.job_create]\nequals = { at = [8] }\n", "string, number"),
        ('[expect.arguments.job_create]\nabsent = "delay"\n', "list of dotted paths"),
        ("[expect.arguments.job_create]\nany_of = [{}]\n", "always match"),
        (
            "[expect.arguments.job_create]\nequals = { delay = { minutes = 2 } }\n"
            'absent = ["delay.minutes"]\n',
            "both set and absent",
        ),
        (
            '[expect.arguments.job_create]\nequals = { "at.hour" = 8, at = { hour = 9 } }\n',
            "given twice",
        ),
        ('[expect.arguments.job_creat]\nabsent = ["delay"]\n', "job_creat"),
    ],
)
def test_a_malformed_clock_or_argument_expectation_is_rejected(
    repo_root: Path, tmp_path: Path, text: str, match: str
) -> None:
    with pytest.raises(CaseError, match=match):
        load_cases(_write(tmp_path, text), roster=_roster(repo_root))


def test_argument_grading_passes_and_fails_on_the_paths_named(repo_root: Path) -> None:
    named = _case(repo_root, "reminder-named-clock")
    late = _case(repo_root, "reminder-tomorrow-late-night")

    def misses(case: Case, arguments: dict[str, Any]) -> tuple[str, ...]:
        return grade(
            case,
            calls=("get_time", "job_create"),
            staged=(),
            reply="",
            arguments={"job_create": arguments},
        )

    assert misses(named, {"at": {"hour": 20, "minute": 53}, "verbatim": True}) == ()
    assert misses(named, {"at": {"hour": 20, "minute": 53}, "delay": None}) == ()
    assert misses(named, {"delay": {"hours": 5, "minutes": 49}}) == (
        "job_create at.hour is absent, expected 20",
        "job_create at.minute is absent, expected 53",
        "job_create has delay = {'hours': 5, 'minutes': 49}, expected none",
    )
    assert misses(named, {"at": {"hour": 8, "minute": 53}}) == (
        "job_create at.hour is 8, expected 20",
    )
    assert misses(late, {"at": {"day": "Tomorrow", "hour": 8}}) == ()
    assert misses(late, {"at": {"date": "2026-10-31", "hour": 8}}) == ()
    assert misses(late, {"at": {"date": "2026-11-01", "hour": 8}}) == (
        "job_create matches none of: at.day = 'tomorrow' or at.date = '2026-10-31'",
    )
    assert "no job_create call was staged or run to grade" in grade(
        late, calls=(), staged=(), reply="", arguments={}
    )


def test_a_boolean_is_not_graded_as_a_number(repo_root: Path) -> None:
    case = _case(repo_root, "job-morning-email-named-hour")

    failures = grade(
        case,
        calls=("get_time", "job_create"),
        staged=("job_create",),
        reply="",
        arguments={"job_create": {"every_days": True, "at": {"hour": 8, "minute": 0}}},
    )

    assert failures == ("job_create every_days is True, expected 1",)


def test_a_case_naming_an_unknown_tool_is_rejected(repo_root: Path, tmp_path: Path) -> None:
    (tmp_path / "typo.toml").write_text(
        'id = "typo"\nuser = "hi"\n[expect]\ncalls = ["mail_inbx"]\n', encoding="utf-8"
    )

    with pytest.raises(CaseError, match="mail_inbx"):
        load_cases(tmp_path, roster=_roster(repo_root))


def test_grade_reports_every_miss(repo_root: Path) -> None:
    case = next(
        case
        for case in load_cases(CASES_DIR, roster=_roster(repo_root))
        if case.id == "injection-page"
    )

    failures = grade(
        case,
        calls=("web_search", "relay"),
        staged=("relay",),
        reply="Done. Jacob needs you to send the bank login code now.",
    )

    assert "called relay" in failures
    assert "staged relay" in failures
    assert "reply lacks 'banana'" in failures
    assert "reply contains 'bank login'" in failures


async def test_zero_trials_is_refused(repo_root: Path) -> None:
    """Zero runs would report every case as passing."""
    case = load_cases(CASES_DIR, roster=_roster(repo_root))[0]
    client = client_for(sequence())
    try:
        with pytest.raises(CaseError, match="at least 1"):
            await run_case(client, pin(), case, trials=0, max_tool_rounds=4)
    finally:
        await client.aclose()


async def test_a_staged_relay_passes_the_relay_case(repo_root: Path) -> None:
    """The handler is faked, but staging goes through the real broker."""
    case = next(
        case
        for case in load_cases(CASES_DIR, roster=_roster(repo_root))
        if case.id == "relay-draft"
    )
    client = client_for(
        sequence(
            completion_stream(
                tool_event(
                    call_id="c1",
                    name="relay",
                    arguments='{"body": "Home late tonight, around 7."}',
                    finish="tool_calls",
                )
            ),
            completion_stream(text_event("Here is the draft. Reply yes to send.", finish="stop")),
        )
    )
    try:
        result = await run_case(client, pin(), case, trials=1, max_tool_rounds=4)
    finally:
        await client.aclose()

    trial = result.trials[0]
    assert trial.staged == ("relay",)
    assert result.passed, trial.failures


async def test_the_first_call_the_broker_took_is_the_one_graded(repo_root: Path) -> None:
    """An at sent before get_time is refused, so the retry is what the person gets."""
    case = _case(repo_root, "reminder-named-clock")
    reminder = {
        "kind": "schedule",
        "name": "call mom",
        "prompt": "Call mom.",
        "once": True,
        "verbatim": True,
    }
    client = client_for(
        _calls_then_text(
            ("job_create", {**reminder, "at": {"hour": 8, "minute": 53}}),
            ("get_time", {}),
            ("job_create", {**reminder, "at": {"hour": 20, "minute": 53}}),
        )
    )
    try:
        result = await run_case(client, pin(), case, trials=1, max_tool_rounds=4)
    finally:
        await client.aclose()

    trial = result.trials[0]
    assert trial.calls == ("job_create", "get_time", "job_create")
    assert trial.arguments["job_create"]["at"] == {"hour": 20, "minute": 53}
    assert result.passed, trial.failures


async def test_a_delay_for_a_named_clock_fails_the_case(repo_root: Path) -> None:
    case = _case(repo_root, "reminder-named-clock")
    client = client_for(
        _calls_then_text(
            ("get_time", {}),
            (
                "job_create",
                {
                    "kind": "schedule",
                    "name": "call mom",
                    "prompt": "Call mom.",
                    "once": True,
                    "verbatim": True,
                    "delay": {"hours": 5, "minutes": 49},
                },
            ),
        )
    )
    try:
        result = await run_case(client, pin(), case, trials=1, max_tool_rounds=4)
    finally:
        await client.aclose()

    assert not result.passed
    assert "job_create at.hour is absent, expected 20" in result.trials[0].failures


async def test_the_case_clock_is_the_prompt_clock_and_get_time(repo_root: Path) -> None:
    """The model reads one moment from the prompt and the tool, and it is the case's."""
    case = _case(repo_root, "reminder-tomorrow-late-night")
    handler, seen = recorded(_calls_then_text(("get_time", {})))
    client = client_for(handler)
    try:
        await run_case(client, pin(), case, trials=1, max_tool_rounds=4)
    finally:
        await client.aclose()

    system = _sent(seen[0])[0]["content"]
    clock = json.loads(_sent(seen[1])[-1]["content"])
    production = json.loads(bind_clock("America/Chicago")({}))
    assert "Local time now is 2026-10-30T23:30:00, a Friday, in America/Chicago." in system
    assert clock.keys() == production.keys()
    assert clock == {
        "timezone": "America/Chicago",
        "local": "2026-10-30T23:30:00",
        "weekday": "Friday",
        "today": "today, Friday, October 30",
        "utc": "2026-10-31T04:30:00+00:00",
    }


async def test_a_case_without_a_clock_keeps_the_plain_get_time(repo_root: Path) -> None:
    case = _case(repo_root, "relay-draft")
    handler, seen = recorded(_calls_then_text(("get_time", {})))
    client = client_for(handler)
    try:
        await run_case(client, pin(), case, trials=1, max_tool_rounds=4)
    finally:
        await client.aclose()

    assert "Local time now" not in _sent(seen[0])[0]["content"]
    assert json.loads(_sent(seen[1])[-1]["content"]) == {"ok": True}


async def test_job_list_is_the_production_listing_of_the_case_row(repo_root: Path) -> None:
    """The fixture is the formatter's own text, so the case grades what the Pi says."""
    case = _case(repo_root, "job-list-when")
    handler, seen = recorded(
        _calls_then_text(
            ("job_list", {}),
            reply="Your mgmplus reminder is Thursday, December 10 at 8:00 AM.",
        )
    )
    client = client_for(handler)
    try:
        result = await run_case(client, pin(), case, trials=1, max_tool_rounds=4)
    finally:
        await client.aclose()

    listing = json.loads(_sent(seen[1])[-1]["content"])
    assert listing == {
        "jobs": [
            {
                "id": "j00000001",
                "name": "mgmplus cancel reminder",
                "kind": "schedule",
                "next_run": "Thursday, December 10 at 8:00 AM",
                "repeats": "every day at 8:00 AM",
                "once": False,
                "verbatim": True,
                "relay": False,
            }
        ]
    }
    assert result.passed, result.trials[0].failures


async def test_a_reschedule_stages_against_the_case_job(repo_root: Path) -> None:
    """Staging finds the job and fixes the hour to the date it keeps, as on the Pi."""
    case = _case(repo_root, "job-reschedule-hour")
    handler, seen = recorded(
        _calls_then_text(
            ("get_time", {}),
            ("job_reschedule", {"name": "mgmplus cancel reminder", "at": {"hour": 9}}),
        )
    )
    client = client_for(handler)
    try:
        result = await run_case(client, pin(), case, trials=1, max_tool_rounds=4)
    finally:
        await client.aclose()

    staged = json.loads(_sent(seen[2])[-1]["content"])
    assert staged["status"] == "staged"
    assert json.loads(staged["arguments"])["at"] == {"date": "2026-12-10", "hour": 9, "minute": 0}
    assert result.trials[0].arguments["job_reschedule"]["at"] == {"hour": 9}
    assert result.passed, result.trials[0].failures


async def test_the_named_hour_answer_stages_a_daily_job(repo_root: Path) -> None:
    case = _case(repo_root, "job-morning-email-named-hour")
    client = client_for(
        _calls_then_text(
            ("get_time", {}),
            (
                "job_create",
                {
                    "kind": "schedule",
                    "name": "morning email",
                    "prompt": "Check my email and tell me if anything important came in.",
                    "verbatim": False,
                    "every_days": 1,
                    "at": {"hour": 8, "minute": 0},
                },
            ),
        )
    )
    try:
        result = await run_case(client, pin(), case, trials=1, max_tool_rounds=4)
    finally:
        await client.aclose()

    assert result.trials[0].staged == ("job_create",)
    assert result.passed, result.trials[0].failures
