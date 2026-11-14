"""Brokered jobs: schedules that always talk, watches that die.

The model does not get cron. It gets named job objects stored in SQLite.
A schedule messages the owner on every run, even when the report is empty.
A watch stays quiet unless it finds something, fails, or hits its TTL.
Jobs report; they never file, draft, write, or compose a relay. A relay
confirmed for later rides on a job row, and the runner sends its stored text.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog

from assistai.clock import (
    AT_SCHEMA,
    DAY_WORDS,
    DELAY_SCHEMA,
    EVERY_DAYS_SCHEMA,
    EVERY_MINUTES_SCHEMA,
    Delay,
    HouseholdClock,
    LocalAt,
    Schedule,
    at_local_of,
    fix_at,
    local_day,
    next_slot,
    resolve_at,
    say_repeat,
    say_when,
)
from assistai.clock import require_representable as _require_in_zone
from assistai.errors import JobError, StoreError
from assistai.inference.types import Message, ToolSpec
from assistai.relay import active_agent, active_taint

log = structlog.get_logger(__name__)

JobKind = Literal["schedule", "watch"]

JOB_CREATE = "job_create"
JOB_LIST = "job_list"
JOB_CANCEL = "job_cancel"
JOB_RESCHEDULE = "job_reschedule"
JOB_TOOLS = (JOB_CREATE, JOB_LIST, JOB_CANCEL, JOB_RESCHEDULE)

DAY_SECONDS = 86400
_DAY_MINUTES = 1440
MAX_TTL_SECONDS = 30 * 24 * 3600
MAX_NAME_CHARS = 80
MAX_PROMPT_CHARS = 2000

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_AT_KEYS = frozenset({"hour", "minute", "day", "date", "fold"})
_DELAY_KEYS = ("days", "hours", "minutes")
# What replaced each pre-016 time argument. A refusal names it, so one retry fixes the call.
_LEGACY_ARGUMENTS = {
    "every_seconds": "every_days with at for whole days, or every_minutes",
    "in_seconds": "delay with days, hours, and minutes",
    "start_at": "at with hour, minute, and day or date",
}
# A run cannot schedule. "Create the … job" is how a December reminder became
# a text every night from October: the date never reached the first-run time.
_ANOTHER_JOB = re.compile(
    r"\b(?:create|make|add|schedule|set up)\b.{0,400}\bjob\b",
    re.IGNORECASE | re.DOTALL,
)
# get_time is not a check. The mail job answered NONE without opening the inbox.
REPORT_TOOLS = frozenset(
    {
        "mail_inbox",
        "mail_read",
        "calendar_today",
        "web_search",
        "web_fetch",
        "shared_lists",
    }
)
_NOTHING = frozenset(
    {
        "",
        "none",
        "nothing",
        "nothing found",
        "nothing to report",
        "nothing_found",
    }
)

ASK_WHAT_TIME = (
    "a daily or weekly job needs the time of day the person named. "
    "Ask them what time, then call again with at"
)

JOB_CREATE_SPEC = ToolSpec(
    name=JOB_CREATE,
    description=(
        "Propose a job for this person only, the one in this chat. "
        "To text the other household member now or later, call relay instead. "
        "Do not refuse that request. "
        "A schedule always texts them "
        "when it runs, even if the report is empty. A watch stays silent unless "
        "it finds something, fails, or expires, and it requires ttl_seconds. "
        "A job run only reports. It does not file, draft, write, or send to the other person. "
        "They may search, fetch, read the shared calendar, and read this person's mail. "
        "Copy the person's words into at: 'tomorrow' is day, a calendar date is date, "
        "and a bare hour is neither. Call get_time in this same turn before using at. "
        "An at without that call is rejected. "
        "Use delay only for a duration they said ('in two minutes', 'in 50 days'). "
        "Do not invent a clock time, and do not turn a clock time into a delay. "
        "A daily or weekly job is every_days with at, and keeps that time of day. "
        "every_minutes is for gaps shorter than a day or not whole days. "
        "If they did not name the time of day for a daily or weekly job, ask them "
        "what time and call no tool until they answer. "
        "Nothing is sent before the first run. "
        "A job cannot create another job. Do not put the date only in the prompt. "
        "If this call is rejected, call get_time if you have not, then call job_create again. "
        "A reminder that begins on December 9 at 8 and then repeats every 10 days is "
        "at with date 2026-12-09 and hour 8, and every_days 10. "
        "once true sends a single message and then the job ends; it has no every_days "
        "or every_minutes and needs at or delay. "
        "verbatim true texts the prompt itself. Use it for a reminder. "
        "It is saved immediately unless this conversation already holds mail, web, "
        "calendar, list, or relay text; then it waits for a yes like any other job. "
        "Do not ask the person to reply yes yourself. "
        "Leave verbatim false when the run should write its own report, such as a mail digest. "
        "That kind still waits for a yes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["schedule", "watch"],
            },
            "name": {
                "type": "string",
                "description": "Short name used later to list or cancel, e.g. morning email.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "For a reminder, the Signal text itself, such as "
                    "'Don't forget to put the cleaner in the pool.' "
                    "For a digest, what to check."
                ),
            },
            "at": AT_SCHEMA,
            "delay": DELAY_SCHEMA,
            "every_days": EVERY_DAYS_SCHEMA,
            "every_minutes": EVERY_MINUTES_SCHEMA,
            "ttl_seconds": {
                "type": "integer",
                "description": "Required for a watch. The watch is cancelled when this elapses.",
            },
            "once": {
                "type": "boolean",
                "description": (
                    "Send one message at the first run, then cancel the job. Set verbatim too."
                ),
            },
            "verbatim": {
                "type": "boolean",
                "description": (
                    "True texts the prompt itself. Required for a reminder, and required "
                    "whenever once is set. False lets the run write a report and waits for a yes."
                ),
            },
        },
        "required": ["kind", "name", "prompt"],
        "additionalProperties": False,
    },
)

JOB_LIST_SPEC = ToolSpec(
    name=JOB_LIST,
    description="List this person's active jobs: name, kind, interval, and when they next run.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)

JOB_CANCEL_SPEC = ToolSpec(
    name=JOB_CANCEL,
    description=(
        "Propose cancelling one of this person's jobs by name or id. "
        "It is cancelled only after they confirm."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Job name as listed."},
            "id": {"type": "string", "description": "Job id if the name is ambiguous."},
        },
        "additionalProperties": False,
    },
)

JOB_RESCHEDULE_SPEC = ToolSpec(
    name=JOB_RESCHEDULE,
    description=(
        "Propose a new time or repeat for an existing job. Give at to move it to a "
        "clock time the person named, copying their words as for job_create, and "
        "call get_time in this same turn first. A bare hour keeps the day the job "
        "was going to run; give day or date only if they named one. Give every_days "
        "or every_minutes to change how often it repeats; that alone leaves the next "
        "run where it is. A daily or weekly repeat keeps its time "
        "of day; a job that does not repeat in days yet needs at to become one, so "
        "ask what time if they did not say. The stored change is what runs after confirm."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Job name as listed."},
            "id": {"type": "string", "description": "Job id if the name is ambiguous."},
            "at": AT_SCHEMA,
            "every_days": EVERY_DAYS_SCHEMA,
            "every_minutes": EVERY_MINUTES_SCHEMA,
        },
        "additionalProperties": False,
    },
)


@dataclass(frozen=True)
class Job:
    """One persisted schedule or watch. Cancelled rows stay for audit."""

    id: str
    agent: str
    kind: JobKind
    name: str
    prompt: str
    every_seconds: int
    ttl_seconds: int | None
    created_at: float
    expires_at: float | None
    next_run_at: float
    last_run_at: float | None
    cancelled_at: float | None
    once: bool = False
    verbatim: bool = False
    relay: bool = False
    relay_sent: bool = False
    relay_recorded: bool = False
    tainted: bool = False
    pending_text: str | None = None
    pending_untrusted: bool = False
    pending_after: Literal["advance", "cancel"] | None = None
    pending_delivered: bool = False
    # The wall-clock schedule (schema 15). None only on a row an older build
    # wrote, until the gateway's next start fills it in.
    zone: str | None = None
    anchor_at: float | None = None
    every_days: int | None = None
    at_local: str | None = None

    def schedule(self) -> Schedule | None:
        """The stored schedule, or None for a row no schema 15 build has filled yet.

        A one-time job has no repeat even though ``every_seconds`` is set:
        the column is required, and an older build reads it after a rollback.
        """
        if self.zone is None or self.anchor_at is None:
            return None
        try:
            zone = ZoneInfo(self.zone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise JobError(f"job {self.id} names an unknown zone") from exc
        if self.once:
            return Schedule(zone=zone, anchor_at=self.anchor_at)
        if self.every_days is not None:
            return Schedule(
                zone=zone,
                anchor_at=self.anchor_at,
                every_days=self.every_days,
                at_local=self.at_local,
            )
        return Schedule(zone=zone, anchor_at=self.anchor_at, every_seconds=self.every_seconds)


def schedule_from_seconds(
    *,
    zone: ZoneInfo,
    first_run: float,
    every_seconds: int,
    once: bool,
) -> Schedule:
    """The wall-clock schedule for a row an older build wrote with only a gap in seconds.

    A whole number of days keeps the clock time of ``first_run``, so a daily
    job holds its hour across daylight saving (decision 0002); any other gap
    is elapsed time. Tool calls name ``every_days`` and ``at`` instead, and
    never come through here.
    """
    if once:
        return Schedule(zone=zone, anchor_at=first_run)
    if every_seconds % DAY_SECONDS == 0:
        return Schedule(
            zone=zone,
            anchor_at=first_run,
            every_days=every_seconds // DAY_SECONDS,
            at_local=at_local_of(first_run, zone),
        )
    return Schedule(zone=zone, anchor_at=first_run, every_seconds=every_seconds)


def schedule_for(
    *,
    zone: ZoneInfo,
    first_run: float,
    once: bool,
    first: LocalAt | Delay | None,
    every_days: int | None,
    every_minutes: int | None,
) -> Schedule:
    """The schedule a parsed create or relay stores.

    A whole-day repeat keeps the clock time the person named in ``at``, not
    the clock time of the instant it resolved to, so nothing reads an hour
    back from a timestamp (decision 0002).
    """
    if once:
        return Schedule(zone=zone, anchor_at=first_run)
    if every_days is not None:
        if not isinstance(first, LocalAt):
            raise JobError(ASK_WHAT_TIME)
        return Schedule(
            zone=zone,
            anchor_at=first_run,
            every_days=every_days,
            at_local=_hhmm(first),
        )
    if every_minutes is not None:
        return Schedule(zone=zone, anchor_at=first_run, every_seconds=every_minutes * 60)
    raise JobError("a repeating job needs every_days or every_minutes")


def first_run_at(
    first: LocalAt | Delay | None, *, clock: HouseholdClock, every_minutes: int | None
) -> float:
    """The instant of the first run, read on the server's clock.

    A named time is resolved in the household zone, a delay counts from now,
    and a minute repeat with neither starts one gap from now.
    """
    if isinstance(first, LocalAt):
        return resolve_at(first, clock=clock)
    if isinstance(first, Delay):
        return clock.now() + first.seconds
    if every_minutes is not None:
        return clock.now() + every_minutes * 60
    raise JobError("the first run needs at or delay")


def upcoming_runs(job: Job, *, count: int, after: float | None = None) -> list[float]:
    """The next ``count`` runs of ``job``, strictly after ``after`` when it is given.

    The stored ``next_run_at`` comes first, even when it is due now, because
    that is the run the runner will fire; the schedule supplies the rest.
    """
    schedule = job.schedule()
    runs: list[float] = []
    cursor = job.next_run_at
    if after is None or job.next_run_at > after:
        runs.append(job.next_run_at)
    else:
        cursor = after
    if schedule is None or not schedule.repeats:
        return runs[:count]
    while len(runs) < count:
        cursor = next_slot(schedule, after=cursor)
        runs.append(cursor)
    return runs[:count]


@dataclass(frozen=True)
class JobOutcome:
    """One report-only run.

    ``found`` is false when the model had nothing to say. ``checked`` is false
    when that empty answer is not a real empty result: no report tool
    returned without an error, or a schedule's inbox listed mail and the
    model still said nothing. A caller that already knows the run looked
    leaves ``checked`` at its default.
    """

    text: str
    found: bool
    untrusted: bool
    checked: bool = True


@dataclass(frozen=True)
class JobCreate:
    """A parsed ``job_create``. ``first`` is when the person said the first run is."""

    kind: JobKind
    name: str
    prompt: str
    ttl_seconds: int | None
    first: LocalAt | Delay | None = None
    every_days: int | None = None
    every_minutes: int | None = None
    once: bool = False
    verbatim: bool = False

    @property
    def every_seconds(self) -> int:
        """The gap written beside the schedule, so an older build can still step the row.

        A one-time job has no gap; the column is required, and a day is what
        older builds stored for one.
        """
        return _gap_seconds(self.every_days, self.every_minutes) or DAY_SECONDS


@dataclass(frozen=True)
class JobReschedule:
    """A parsed ``job_reschedule``: a new time, a new repeat, or both."""

    name: str | None
    id: str | None
    at: LocalAt | None = None
    every_days: int | None = None
    every_minutes: int | None = None

    @property
    def gap_seconds(self) -> int | None:
        return _gap_seconds(self.every_days, self.every_minutes)


def new_job_id() -> str:
    return "j" + secrets.token_hex(4)


def legacy_time_arguments(arguments: dict[str, Any]) -> list[str]:
    """The pre-016 time arguments a call still carries, in a fixed order."""
    return [name for name in _LEGACY_ARGUMENTS if arguments.get(name) is not None]


def parse_create(arguments: dict[str, Any]) -> JobCreate:
    """Every rule a job's time arguments must meet, on every path that creates one.

    A digest stages, a clean reminder saves at once, and a relay stores a
    job row, and all three call this first, so a refusal here means nothing
    is staged or saved. A whole-day repeat with no named hour is refused:
    "every morning" gets "what time?" back, not a job at the minute it was
    asked.
    """
    _refuse_legacy(arguments)
    kind = arguments.get("kind")
    if kind not in ("schedule", "watch"):
        raise JobError("kind must be schedule or watch")
    name = _parse_name(arguments.get("name"))
    prompt = _parse_prompt(arguments.get("prompt"))
    once = _parse_bool(arguments.get("once"), default=False, field="once")
    verbatim_raw = arguments.get("verbatim")
    verbatim = _parse_bool(verbatim_raw, default=False, field="verbatim")
    if kind == "watch" and (once or verbatim):
        raise JobError("a watch cannot be a reminder")
    first = _parse_first(arguments)
    every_days, every_minutes = _parse_repeat(arguments)
    if every_days is not None and not isinstance(first, LocalAt):
        # A delay says when the first run is, not the hour it repeats at.
        raise JobError(ASK_WHAT_TIME)
    if once and (every_days is not None or every_minutes is not None):
        raise JobError(
            "a one-time job does not repeat. Drop every_days and every_minutes, or set once false"
        )
    if not once and every_days is None and every_minutes is None:
        raise JobError(
            "a repeating job needs every_days with at, or every_minutes. "
            "For a single text, set once true"
        )
    if once and verbatim_raw is None:
        # Guessing either way is wrong: a reminder taken for a digest texts a
        # report, and a digest taken for a reminder skips the yes.
        raise JobError(
            "a one-time job must set verbatim: true to text this sentence, "
            "or false for a one-time report"
        )
    if once and first is None:
        raise JobError("a one-time job needs at or delay for its run")
    ttl_seconds = _parse_ttl(
        arguments.get("ttl_seconds"),
        kind=kind,
        every_seconds=_gap_seconds(every_days, every_minutes) or 0,
    )
    # A verbatim prompt is texted, never run, so it cannot schedule anything.
    if not verbatim and _asks_for_another_job(prompt):
        raise JobError(
            "a job cannot create another job. Schedule the reminder itself: "
            "verbatim true, the Signal sentence as the prompt, and at or delay "
            "for the first text. every_days or every_minutes is the gap after that. "
            "Nothing is sent before the first text."
        )
    if verbatim and first is None:
        raise JobError(
            "a reminder needs at or delay for the first text. "
            "every_days or every_minutes is only the gap after that. "
            "Nothing is sent before then."
        )
    return JobCreate(
        kind=kind,
        name=name,
        prompt=prompt,
        ttl_seconds=ttl_seconds,
        first=first,
        every_days=every_days,
        every_minutes=every_minutes,
        once=once,
        verbatim=verbatim,
    )


def parse_reschedule(arguments: dict[str, Any]) -> JobReschedule:
    _refuse_legacy(arguments)
    name = arguments.get("name")
    job_id = arguments.get("id")
    parsed_name = _parse_name(name) if name is not None else None
    parsed_id = _parse_id(job_id) if job_id is not None else None
    if parsed_name is None and parsed_id is None:
        raise JobError("reschedule needs a name or id")
    at = parse_at(arguments["at"]) if arguments.get("at") is not None else None
    every_days, every_minutes = _parse_repeat(arguments)
    if at is None and every_days is None and every_minutes is None:
        raise JobError("reschedule needs at, every_days, or every_minutes")
    return JobReschedule(
        name=parsed_name,
        id=parsed_id,
        at=at,
        every_days=every_days,
        every_minutes=every_minutes,
    )


def check_reschedule(job: Job, spec: JobReschedule) -> None:
    """Refuse a reschedule the job cannot take, before it is staged and again when it runs."""
    gap = spec.gap_seconds
    if job.ttl_seconds is not None and gap is not None and gap > job.ttl_seconds:
        raise JobError("the new repeat is longer than the watch TTL")
    if job.once and gap is not None:
        raise JobError("a one-time job does not repeat. Give at to move it")
    if spec.every_days is not None and spec.at is None and job.every_days is None:
        raise JobError(ASK_WHAT_TIME)


def parse_at(raw: object) -> LocalAt:
    """A named clock time as the model sent it. Resolved later, on the server's clock."""
    if not isinstance(raw, dict):
        raise JobError("at must be an object with hour, minute, and day or date")
    unknown = sorted(str(key) for key in raw if key not in _AT_KEYS)
    if unknown:
        raise JobError(f"at takes hour, minute, and day or date, not {', '.join(unknown)}")
    hour = _whole(raw.get("hour"), field="at.hour", minimum=0, maximum=23)
    minute_raw = raw.get("minute")
    minute = 0 if minute_raw is None else _whole(minute_raw, field="at.minute", maximum=59)
    day_raw, date_raw = raw.get("day"), raw.get("date")
    if day_raw is not None and date_raw is not None:
        raise JobError("at names a day or a date, not both")
    day: str | None = None
    if day_raw is not None:
        if not isinstance(day_raw, str) or day_raw.strip().lower() not in DAY_WORDS:
            raise JobError("at.day must be today, tomorrow, or a weekday name")
        day = day_raw.strip().lower()
    date: dt.date | None = None
    if date_raw is not None:
        if not isinstance(date_raw, str) or not _DATE.fullmatch(date_raw.strip()):
            raise JobError("at.date must be YYYY-MM-DD")
        try:
            date = dt.date.fromisoformat(date_raw.strip())
        except ValueError as exc:
            raise JobError("at.date is not a calendar date") from exc
    fold_raw = raw.get("fold")
    fold = None if fold_raw is None else _whole(fold_raw, field="at.fold", maximum=1)
    return LocalAt(date=date, day=day, hour=hour, minute=minute, fold=fold)


def parse_delay(raw: object) -> Delay:
    """A spoken duration. It counts from when the call runs, which for a staged one is the yes."""
    if not isinstance(raw, dict):
        raise JobError("delay must be an object with days, hours, and minutes")
    unknown = sorted(str(key) for key in raw if key not in _DELAY_KEYS)
    if unknown:
        raise JobError(f"delay takes days, hours, and minutes, not {', '.join(unknown)}")
    parts = {
        key: 0 if raw.get(key) is None else _whole(raw.get(key), field=f"delay.{key}")
        for key in _DELAY_KEYS
    }
    delay = Delay(days=parts["days"], hours=parts["hours"], minutes=parts["minutes"])
    if delay.seconds < 60:
        raise JobError("delay must be at least one minute")
    return delay


def reschedule_day(job: Job, *, clock: HouseholdClock) -> dt.date | None:
    """The date a bare ``at`` keeps when it moves ``job``, or None to read it as job_create does.

    Moving a daily or one-time job to another hour keeps the day it was
    going to run on, so "move it to 9" does not pull the December 7
    reminder to tomorrow. A minute repeat has no day of its own, so a bare
    hour there is the next time the clock shows it.
    """
    if job.every_days is None and not job.once:
        return None
    return local_day(job.next_run_at, clock.zone)


def resolved_arguments(
    arguments: dict[str, Any], *, clock: HouseholdClock, job: Job | None = None
) -> dict[str, Any]:
    """The call as it is staged: a relative ``at`` fixed to the date it means now.

    The preview is shown now and the yes runs later, so "tomorrow" and a
    bare hour are stored as the date the person was shown, plus ``fold``
    for an hour the clocks repeat. A reschedule passes the ``job`` it
    moves, so a bare hour is fixed to the date that job keeps, the same
    date its handler would pick. Only ``at`` changes, and only when the
    model sent one. A delay is left alone: it counts from the yes, and the
    preview says so.
    """
    raw = arguments.get("at")
    if raw is None:
        return arguments
    on = reschedule_day(job, clock=clock) if job is not None else None
    fixed = fix_at(parse_at(raw), clock=clock, on=on)
    if fixed.date is None:
        raise JobError("at could not be fixed to a date")
    stored: dict[str, object] = {
        "date": fixed.date.isoformat(),
        "hour": fixed.hour,
        "minute": fixed.minute,
    }
    if fixed.fold is not None:
        stored["fold"] = fixed.fold
    return {**arguments, "at": stored}


def parse_cancel(arguments: dict[str, Any]) -> tuple[str | None, str | None]:
    name = arguments.get("name")
    job_id = arguments.get("id")
    parsed_name = _parse_name(name) if name is not None else None
    parsed_id = _parse_id(job_id) if job_id is not None else None
    if parsed_name is None and parsed_id is None:
        raise JobError("cancel needs a name or id")
    return parsed_name, parsed_id


def is_nothing(text: str) -> bool:
    """Watches stay quiet on these; schedules still send a placeholder.

    The run prompt asks for the single word NONE, and models punctuate it, so
    trailing marks are ignored. ``no`` is left alone: it is a real answer.
    """
    return text.strip().rstrip(".!").strip().lower() in _NOTHING


def empty_report_is_real(
    messages: Sequence[Message], *, start: int, kind: JobKind = "schedule"
) -> bool:
    """True when NONE follows a look that found nothing.

    The morning mail job answered NONE in two tokens. That is not an empty
    inbox, and neither is an inbox call that came back as an error. For a
    schedule, an inbox that listed mail and still got NONE is the same miss.
    A watch is different: one waiting for a single sender sees other mail
    and is right to say NONE.
    """
    looks = _report_looks(messages, start=start)
    if not looks.succeeded:
        return False
    return kind == "watch" or not looks.listed_mail


def look_again_prompt(
    messages: Sequence[Message], *, start: int, kind: JobKind = "schedule"
) -> str:
    """One nudge when the run answered before it had a real result."""
    looks = _report_looks(messages, start=start)
    if looks.succeeded and looks.listed_mail and kind == "schedule":
        return (
            "A tool already returned something to report. "
            "Summarize that result. Do not answer NONE."
        )
    if looks.attempted and not looks.succeeded:
        return (
            "The tool returned an error, so nothing was checked. Call it once more. "
            "If it fails again, say what failed. Do not answer NONE."
        )
    return (
        "You answered without looking. Call a tool now, then report what it returned. "
        "NONE is only valid after a tool result shows there is nothing."
    )


@dataclass(frozen=True)
class _Looks:
    attempted: bool
    succeeded: bool
    listed_mail: bool


def _report_looks(messages: Sequence[Message], *, start: int) -> _Looks:
    """What the report tools in this run actually returned."""
    report_calls: set[str] = set()
    attempted = False
    succeeded = False
    listed_mail = False
    for message in messages[start:]:
        if message.role == "assistant":
            for call in message.tool_calls:
                if call.name in REPORT_TOOLS:
                    report_calls.add(call.id)
                    attempted = True
        elif message.role == "tool" and message.tool_call_id in report_calls:
            if _tool_errored(message.content):
                continue
            succeeded = True
            listed_mail = listed_mail or _tool_listed_mail(message.content)
    return _Looks(attempted=attempted, succeeded=succeeded, listed_mail=listed_mail)


def _tool_errored(content: str | None) -> bool:
    if not content:
        return False
    try:
        parsed: object = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and "error" in parsed


def _asks_for_another_job(prompt: str) -> bool:
    """A run cannot schedule. The first text belongs in at or delay."""
    if "job_create" in prompt.casefold():
        return True
    return _ANOTHER_JOB.search(prompt) is not None


def _tool_listed_mail(content: str | None) -> bool:
    if not content:
        return False
    try:
        parsed: object = json.loads(content)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    unread = parsed.get("inbox_unread")
    if isinstance(unread, int) and not isinstance(unread, bool) and unread > 0:
        return True
    rows = parsed.get("messages")
    return isinstance(rows, list) and len(rows) > 0


def format_job_message(name: str, body: str) -> str:
    return f"[job: {name}]\n{body}"


def format_expire_message(name: str) -> str:
    return f"[job: {name}]\nThis watch ended. Nothing further will run unless you create it again."


def format_fail_message(name: str) -> str:
    return f"[job: {name}]\nThis check failed. I could not complete it."


def job_user_prompt(job: Job) -> str:
    return (
        f"Scheduled job '{job.name}': {job.prompt}\n\n"
        "Report only. Do not create jobs, relay, file, draft, or propose actions. "
        "You may search, fetch, read the shared calendar, read lists visible "
        "to this person, and read this person's mail; cite URLs. Do not change "
        "a list. Another person's private list is not visible to you. "
        "Call a tool and report what it returned. "
        "Reply with the single word NONE only after a tool result shows there is nothing."
    )


def household_clock(timezone: str) -> HouseholdClock:
    """The household clock for handlers bound with a zone name.

    It reads ``_now`` on every call, not once, so the clock a handler sees
    is the one the job module reads everywhere else.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise JobError("the household timezone is not configured") from exc
    return HouseholdClock(zone=zone, now=current_time)


def bind_jobs(
    store: Any, *, timezone: str = "UTC"
) -> dict[str, Callable[[dict[str, Any]], Awaitable[str]]]:
    """Handlers closed over the live store. Bound when the channel starts."""

    async def create(arguments: dict[str, Any]) -> str:
        spec = parse_create(arguments)
        agent = active_agent()
        clock = household_clock(timezone)
        now = clock.now()
        if store.find_job(agent.name, name=spec.name) is not None:
            raise JobError(f"a job named {spec.name!r} already exists")
        first_run = first_run_at(spec.first, clock=clock, every_minutes=spec.every_minutes)
        # Reject a delay the clock cannot represent before the row exists.
        # Saving first and then failing the reply leaves an armed reminder.
        require_representable(first_run, timezone)
        schedule = schedule_for(
            zone=clock.zone,
            first_run=first_run,
            once=spec.once,
            first=spec.first,
            every_days=spec.every_days,
            every_minutes=spec.every_minutes,
        )
        expires_at = None
        if spec.ttl_seconds is not None:
            origin = first_run if spec.first is not None else now
            expires_at = origin + spec.ttl_seconds
        saved: Job | None = None
        last_error: StoreError | None = None
        for _ in range(5):
            job = Job(
                id=new_job_id(),
                agent=agent.name,
                kind=spec.kind,
                name=spec.name,
                prompt=spec.prompt,
                every_seconds=spec.every_seconds,
                ttl_seconds=spec.ttl_seconds,
                created_at=now,
                expires_at=expires_at,
                next_run_at=first_run,
                last_run_at=None,
                cancelled_at=None,
                once=spec.once,
                verbatim=spec.verbatim,
                tainted=active_taint(),
                zone=schedule.zone.key,
                anchor_at=schedule.anchor_at,
                every_days=schedule.every_days,
                at_local=schedule.at_local,
            )
            try:
                store.save_job(job)
            except StoreError as exc:
                last_error = exc
                if store.find_job(agent.name, name=spec.name) is not None:
                    raise JobError(f"a job named {spec.name!r} already exists") from exc
                continue
            saved = job
            break
        if saved is None:
            raise last_error if last_error is not None else StoreError("job could not be saved")
        job = saved
        words = _schedule_in_words(job, clock=clock)
        payload: dict[str, object] = {
            "ok": True,
            "id": job.id,
            "name": job.name,
            "kind": job.kind,
            **_repeat_fields(spec.every_days, spec.every_minutes),
            "first_run": say_when(job.next_run_at, clock=clock),
            "repeats": say_repeat(words, clock=clock),
            "once": job.once,
            "verbatim": job.verbatim,
        }
        if job.expires_at is not None:
            payload["ends"] = say_when(job.expires_at, clock=clock)
        if job.verbatim:
            payload["say"] = describe_saved_reminder(
                prompt=job.prompt,
                fires_at=job.next_run_at,
                clock=clock,
                name=job.name,
                repeats=say_repeat(words, clock=clock) if words.repeats else None,
            )
            payload["note"] = (
                "The person is sent the say sentence ahead of your reply. "
                "Do not restate the reminder or its time; answer anything else they asked."
            )
        else:
            payload["say"] = describe_saved_job(job, clock=clock)
        return json.dumps(payload, ensure_ascii=False)

    async def list_jobs(_arguments: dict[str, Any]) -> str:
        agent = active_agent()
        clock = household_clock(timezone)
        jobs = store.list_jobs(agent.name)
        payload: list[dict[str, object]] = []
        for job in jobs:
            entry: dict[str, object] = {
                "id": job.id,
                "name": job.name,
                "kind": job.kind,
                "next_run": say_when(job.next_run_at, clock=clock),
                "repeats": say_repeat(_schedule_in_words(job, clock=clock), clock=clock),
                "once": job.once,
                "verbatim": job.verbatim,
                "relay": job.relay,
            }
            if job.expires_at is not None:
                entry["ends"] = say_when(job.expires_at, clock=clock)
            payload.append(entry)
        return json.dumps({"jobs": payload}, ensure_ascii=False)

    async def cancel(arguments: dict[str, Any]) -> str:
        agent = active_agent()
        name, job_id = parse_cancel(arguments)
        job = store.find_job(agent.name, name=name, job_id=job_id)
        if job is None:
            raise JobError("no matching job")
        clock = household_clock(timezone)
        store.cancel_job(job.id, at=clock.now())
        return json.dumps(
            {
                "ok": True,
                "id": job.id,
                "name": job.name,
                "say": (
                    f"Cancelled {job.name!r}. "
                    f"Its next run, {say_when(job.next_run_at, clock=clock)}, will not happen."
                ),
            },
            ensure_ascii=False,
        )

    async def reschedule(arguments: dict[str, Any]) -> str:
        spec = parse_reschedule(arguments)
        agent = active_agent()
        job = store.find_job(agent.name, name=spec.name, job_id=spec.id)
        if job is None:
            raise JobError("no matching job")
        check_reschedule(job, spec)
        clock = household_clock(timezone)
        schedule, every_seconds = rescheduled(job, spec, clock=clock)
        next_run_at = schedule.anchor_at
        require_representable(next_run_at, timezone)
        if not store.reschedule_job(
            job.id,
            every_seconds=every_seconds,
            next_run_at=next_run_at,
            schedule=schedule,
        ):
            raise JobError("no matching job")
        next_run = say_when(next_run_at, clock=clock)
        repeats = say_repeat(schedule, clock=clock)
        then = f", then {repeats}" if schedule.repeats else ""
        return json.dumps(
            {
                "ok": True,
                "id": job.id,
                "name": job.name,
                **_repeat_fields(schedule.every_days, _minutes_of(schedule)),
                "next_run": next_run,
                "repeats": repeats,
                "say": f"Moved {job.name!r}: next run {next_run}{then}.",
            },
            ensure_ascii=False,
        )

    return {
        JOB_CREATE: create,
        JOB_LIST: list_jobs,
        JOB_CANCEL: cancel,
        JOB_RESCHEDULE: reschedule,
    }


def rescheduled(job: Job, spec: JobReschedule, *, clock: HouseholdClock) -> tuple[Schedule, int]:
    """The schedule a reschedule stores, and the gap written beside it.

    The preview calls this too, so the next run it names is the one the
    yes stores.

    Only a new repeat leaves the next run where it is and re-anchors the
    grid on it, so moving the December 7 reminder to weekly keeps December 7
    at 8:00 AM (C-3). A new ``at`` is the next run: a bare hour keeps the
    date ``reschedule_day`` names. A whole-day repeat keeps the hour it has
    unless ``at`` names a new one, and a job becomes whole-day only with an
    ``at``, as in job_create.
    """
    zone = clock.zone
    if spec.at is not None:
        first = resolve_at(spec.at, clock=clock, on=reschedule_day(job, clock=clock))
    else:
        first = job.next_run_at
    if job.once:
        if spec.at is None:
            raise JobError("a one-time job does not repeat. Give at to move it")
        return Schedule(zone=zone, anchor_at=first), job.every_seconds
    every_days = spec.every_days
    every_seconds = spec.every_minutes * 60 if spec.every_minutes is not None else None
    if every_days is None and every_seconds is None:
        every_days = job.every_days
        every_seconds = job.every_seconds if job.every_days is None else None
    if every_days is not None:
        at_local = _hhmm(spec.at) if spec.at is not None else job.at_local
        if at_local is None:
            raise JobError(ASK_WHAT_TIME)
        schedule = Schedule(zone=zone, anchor_at=first, every_days=every_days, at_local=at_local)
        return schedule, every_days * DAY_SECONDS
    if every_seconds is None:
        raise JobError("a repeating job needs every_days or every_minutes")
    return Schedule(zone=zone, anchor_at=first, every_seconds=every_seconds), every_seconds


def _repeat_fields(every_days: int | None, every_minutes: int | None) -> dict[str, object]:
    """The repeat in the argument names the model sends, so a result never teaches an old one."""
    if every_days is not None:
        return {"every_days": every_days}
    if every_minutes is not None:
        return {"every_minutes": every_minutes}
    return {}


def _minutes_of(schedule: Schedule) -> int | None:
    if schedule.every_seconds is None:
        return None
    return schedule.every_seconds // 60


def _parse_name(raw: object) -> str:
    if not isinstance(raw, str):
        raise JobError("name must be a string")
    name = " ".join(raw.split())
    if not _NAME.fullmatch(name):
        raise JobError("name is not a usable job title")
    if len(name) > MAX_NAME_CHARS:
        raise JobError("name is too long")
    return name


def _parse_prompt(raw: object) -> str:
    if not isinstance(raw, str):
        raise JobError("prompt must be a string")
    prompt = raw.strip()
    if not prompt:
        raise JobError("prompt is empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise JobError("prompt is too long")
    return prompt


def current_time() -> float:
    return _now()


def describe_saved_reminder(
    *,
    prompt: str,
    fires_at: float,
    clock: HouseholdClock,
    audience: str = "you",
    name: str | None = None,
    repeats: str | None = None,
) -> str:
    """The sentence the person sees. The clock is the server's, not the model's.

    The time is spelled "8:00 AM" so the reply check, which drops a model
    answer naming any other clock time, reads the same time here.
    """
    require_representable(fires_at, clock.zone.key)
    when = say_when(fires_at, clock=clock)
    then = f", then {repeats}" if repeats else ""
    sentence = f'I will text {audience} {when}{then}: "{prompt}"'
    if name:
        return f"{sentence}\nName: {name}"
    return sentence


def describe_saved_job(job: Job, *, clock: HouseholdClock) -> str:
    """What a confirmed digest or watch will do, in the server's words: its first run and repeat."""
    kind = "schedule" if job.kind == "schedule" else "watch"
    words = _schedule_in_words(job, clock=clock)
    then = f", then {say_repeat(words, clock=clock)}" if words.repeats else ", once"
    ends = f", ending {say_when(job.expires_at, clock=clock)}" if job.expires_at is not None else ""
    return (
        f"Created {kind} {job.name!r}: first run "
        f"{say_when(job.next_run_at, clock=clock)}{then}{ends}."
    )


def require_representable(moment: float, timezone: str) -> None:
    """Fail before a save when the instant cannot be shown in the household zone."""
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise JobError("the household timezone is not configured") from exc
    _require_in_zone(moment, zone)


def _schedule_in_words(job: Job, *, clock: HouseholdClock) -> Schedule:
    """The schedule a job is described by, read from its gap when no build has stored one yet.

    A row an older image wrote is described the way the next start will
    fill it in, so a listing never falls back to a number of seconds.
    """
    try:
        schedule = job.schedule()
    except JobError:
        schedule = None
    if schedule is not None:
        return schedule
    return schedule_from_seconds(
        zone=clock.zone,
        first_run=job.next_run_at,
        every_seconds=job.every_seconds,
        once=job.once,
    )


def _parse_ttl(raw: object, *, kind: JobKind, every_seconds: int) -> int | None:
    if kind == "schedule":
        if raw is None:
            return None
        raise JobError("a schedule has no TTL; cancel it instead")
    if raw is None:
        raise JobError("a watch requires ttl_seconds")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise JobError("ttl_seconds must be an integer")
    value = int(raw)
    if value != raw:
        raise JobError("ttl_seconds must be an integer")
    if value < every_seconds or value > MAX_TTL_SECONDS:
        raise JobError("ttl_seconds is out of range")
    return value


def _parse_bool(raw: object, *, default: bool, field: str) -> bool:
    if raw is None:
        return default
    if not isinstance(raw, bool):
        raise JobError(f"{field} must be true or false")
    return raw


def _refuse_legacy(arguments: dict[str, Any]) -> None:
    """Old calls stay in history for two weeks and the model may copy them. One retry fixes it."""
    found = legacy_time_arguments(arguments)
    if not found:
        return
    log.warning("jobs.legacy_arguments_refused", fields=found)
    replaced = "; ".join(f"instead of {name}, use {_LEGACY_ARGUMENTS[name]}" for name in found)
    raise JobError(
        f"{', '.join(found)} {'is' if len(found) == 1 else 'are'} no longer accepted: "
        f"{replaced}. Call again with those."
    )


def _parse_first(arguments: dict[str, Any]) -> LocalAt | Delay | None:
    at_raw, delay_raw = arguments.get("at"), arguments.get("delay")
    if at_raw is not None and delay_raw is not None:
        raise JobError("set at or delay, not both")
    if at_raw is not None:
        return parse_at(at_raw)
    if delay_raw is not None:
        return parse_delay(delay_raw)
    return None


def _parse_repeat(arguments: dict[str, Any]) -> tuple[int | None, int | None]:
    days_raw, minutes_raw = arguments.get("every_days"), arguments.get("every_minutes")
    if days_raw is not None and minutes_raw is not None:
        raise JobError("set every_days or every_minutes, not both")
    every_days = None if days_raw is None else _whole(days_raw, field="every_days", minimum=1)
    every_minutes = (
        None if minutes_raw is None else _whole(minutes_raw, field="every_minutes", minimum=1)
    )
    if every_minutes is not None and every_minutes % _DAY_MINUTES == 0:
        # A whole day in minutes is elapsed time: it would drift an hour at
        # each daylight saving change, and it would carry no named hour.
        raise JobError(
            f"every_minutes {every_minutes} is whole days: "
            "use every_days so it keeps its local time"
        )
    return every_days, every_minutes


def _whole(raw: object, *, field: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise JobError(f"{field} must be a whole number")
    value = int(raw)
    if value != raw:
        raise JobError(f"{field} must be a whole number")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
        raise JobError(f"{field} must be {bound}")
    return value


def _gap_seconds(every_days: int | None, every_minutes: int | None) -> int | None:
    if every_days is not None:
        return every_days * DAY_SECONDS
    if every_minutes is not None:
        return every_minutes * 60
    return None


def _hhmm(at: LocalAt) -> str:
    return f"{at.hour:02d}:{at.minute:02d}"


def _parse_id(raw: object) -> str:
    if not isinstance(raw, str) or not re.fullmatch(r"j[0-9a-f]{8}", raw.strip()):
        raise JobError("id is not a job id")
    return raw.strip()


def _now() -> float:
    return time.time()
