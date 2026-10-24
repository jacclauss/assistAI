"""Brokered jobs: schedules that always talk, watches that die.

The model does not get cron. It gets named job objects stored in SQLite.
A schedule messages the owner on every run, even when the report is empty.
A watch stays quiet unless it finds something, fails, or hits its TTL.
Jobs report; they never file, draft, write, or compose a relay. A relay
confirmed for later rides on a job row, and the runner sends its stored text.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from assistai.errors import JobError, StoreError
from assistai.inference.types import Message, ToolSpec
from assistai.relay import active_agent, active_taint

JobKind = Literal["schedule", "watch"]

JOB_CREATE = "job_create"
JOB_LIST = "job_list"
JOB_CANCEL = "job_cancel"
JOB_RESCHEDULE = "job_reschedule"
JOB_TOOLS = (JOB_CREATE, JOB_LIST, JOB_CANCEL, JOB_RESCHEDULE)

MIN_EVERY_SECONDS = 60
MAX_TTL_SECONDS = 30 * 24 * 3600
_START_SKEW_SECONDS = 60.0
MAX_NAME_CHARS = 80
MAX_PROMPT_CHARS = 2000

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")
# A run cannot schedule. "Create the … job" is how a December reminder became
# a text every night from October: the date never reached start_at.
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

JOB_CREATE_SPEC = ToolSpec(
    name=JOB_CREATE,
    description=(
        "Propose a repeating job for this person only, the one in this chat. "
        "To text the other household member now or later, call relay instead. "
        "Do not refuse that request. "
        "A schedule always texts them "
        "when it runs, even if the report is empty. A watch stays silent unless "
        "it finds something, fails, or expires, and it requires ttl_seconds. "
        "A job run only reports. It does not file, draft, write, or send to the other person. "
        "They may search, fetch, read the shared calendar, and read this person's mail. "
        "every_seconds is how often it repeats after the first run (86400 for daily, "
        "or any larger gap). It is at least 60 seconds and has no maximum. "
        "Nothing is sent before that first run. "
        "A job cannot create another job. Do not put the date only in the prompt. "
        "For a delay ('in two minutes', 'in 50 days'), set in_seconds and omit start_at. "
        "The server adds that many seconds to the current time. Do not invent a clock time. "
        "start_at is a named local time, ISO-8601 without a timezone suffix "
        "(2026-12-07T08:00:00). Call get_time in this same turn before using it. "
        "A start_at without that call is rejected. "
        "A reminder that begins in 50 days and then repeats every 10 days is "
        "in_seconds 4320000 and every_seconds 864000. "
        "once true sends a single message and then the job ends. "
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
            "every_seconds": {
                "type": "integer",
                "description": (
                    "Seconds between runs after the first. 86400 is daily. "
                    "Any longer gap is allowed. At least 60."
                ),
            },
            "ttl_seconds": {
                "type": "integer",
                "description": "Required for a watch. The watch is cancelled when this elapses.",
            },
            "in_seconds": {
                "type": "integer",
                "description": (
                    "Delay until the first run, in seconds, measured by the server. "
                    "Use this for 'in two minutes' or 'in 50 days'. At least 60. "
                    "Omit start_at when you set this."
                ),
            },
            "start_at": {
                "type": "string",
                "description": (
                    "Named first run, local ISO-8601 without a timezone suffix. "
                    "Requires get_time in this turn. Omit when using in_seconds."
                ),
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
        "required": ["kind", "name", "prompt", "every_seconds"],
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
        "Propose a new interval for an existing job. "
        "The stored interval is what runs after confirm."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Job name as listed."},
            "id": {"type": "string", "description": "Job id if the name is ambiguous."},
            "every_seconds": {
                "type": "integer",
                "description": "New seconds between runs.",
            },
        },
        "required": ["every_seconds"],
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
    kind: JobKind
    name: str
    prompt: str
    every_seconds: int
    ttl_seconds: int | None
    start_at: str | None = None
    in_seconds: int | None = None
    once: bool = False
    verbatim: bool = False


@dataclass(frozen=True)
class JobReschedule:
    every_seconds: int
    name: str | None
    id: str | None


def new_job_id() -> str:
    return "j" + secrets.token_hex(4)


def parse_create(arguments: dict[str, Any]) -> JobCreate:
    kind = arguments.get("kind")
    if kind not in ("schedule", "watch"):
        raise JobError("kind must be schedule or watch")
    name = _parse_name(arguments.get("name"))
    prompt = _parse_prompt(arguments.get("prompt"))
    every_seconds = _parse_every(arguments.get("every_seconds"))
    ttl_seconds = _parse_ttl(arguments.get("ttl_seconds"), kind=kind, every_seconds=every_seconds)
    once = _parse_bool(arguments.get("once"), default=False, field="once")
    verbatim_raw = arguments.get("verbatim")
    verbatim = _parse_bool(verbatim_raw, default=False, field="verbatim")
    if kind == "watch" and (once or verbatim):
        raise JobError("a watch cannot be a reminder")
    start_at = _parse_start_text(arguments.get("start_at"))
    in_seconds = _parse_delay(arguments.get("in_seconds"))
    if start_at is not None and in_seconds is not None:
        raise JobError("set in_seconds or start_at, not both")
    if once and verbatim_raw is None:
        # Guessing either way is wrong: a reminder taken for a digest texts a
        # report, and a digest taken for a reminder skips the yes.
        raise JobError(
            "a one-time job must set verbatim: true to text this sentence, "
            "or false for a one-time report"
        )
    # A verbatim prompt is texted, never run, so it cannot schedule anything.
    if not verbatim and _asks_for_another_job(prompt):
        raise JobError(
            "a job cannot create another job. Schedule the reminder itself: "
            "verbatim true, the Signal sentence as the prompt, and in_seconds "
            "or start_at for the first text. every_seconds is the gap after that. "
            "Nothing is sent before the first text."
        )
    if verbatim and start_at is None and in_seconds is None:
        raise JobError(
            "a reminder needs in_seconds or start_at for the first text. "
            "every_seconds is only the gap after that. Nothing is sent before then."
        )
    return JobCreate(
        kind=kind,
        name=name,
        prompt=prompt,
        every_seconds=every_seconds,
        ttl_seconds=ttl_seconds,
        start_at=start_at,
        in_seconds=in_seconds,
        once=once,
        verbatim=verbatim,
    )


def parse_reschedule(arguments: dict[str, Any]) -> JobReschedule:
    every_seconds = _parse_every(arguments.get("every_seconds"))
    name = arguments.get("name")
    job_id = arguments.get("id")
    parsed_name = _parse_name(name) if name is not None else None
    parsed_id = _parse_id(job_id) if job_id is not None else None
    if parsed_name is None and parsed_id is None:
        raise JobError("reschedule needs a name or id")
    return JobReschedule(every_seconds=every_seconds, name=parsed_name, id=parsed_id)


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
    """A run cannot schedule. The first text belongs in start_at or in_seconds."""
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


def resolve_start_at(text: str, *, now: float, timezone: str) -> float:
    """Unix time for a local ISO-8601 start. A timezone suffix is rejected."""
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise JobError("the household timezone is not configured") from exc
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise JobError("start_at must be an ISO-8601 date and time") from exc
    if parsed.tzinfo is not None:
        raise JobError("start_at must not include a timezone suffix")
    parsed = parsed.replace(tzinfo=zone)
    when = parsed.timestamp()
    if when < now - _START_SKEW_SECONDS:
        raise JobError("start_at is in the past")
    return when


def bind_jobs(
    store: Any, *, timezone: str = "UTC"
) -> dict[str, Callable[[dict[str, Any]], Awaitable[str]]]:
    """Handlers closed over the live store. Bound when the channel starts."""

    async def create(arguments: dict[str, Any]) -> str:
        spec = parse_create(arguments)
        agent = active_agent()
        now = _now()
        if store.find_job(agent.name, name=spec.name) is not None:
            raise JobError(f"a job named {spec.name!r} already exists")
        if spec.in_seconds is not None:
            first_run = now + spec.in_seconds
        elif spec.start_at is not None:
            first_run = resolve_start_at(spec.start_at, now=now, timezone=timezone)
        else:
            first_run = now + spec.every_seconds
        # Reject a delay the clock cannot represent before the row exists.
        # Saving first and then failing the reply leaves an armed reminder.
        require_representable(first_run, timezone)
        expires_at = None
        if spec.ttl_seconds is not None:
            origin = first_run if spec.start_at is not None or spec.in_seconds is not None else now
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
        payload: dict[str, object] = {
            "ok": True,
            "id": job.id,
            "name": job.name,
            "kind": job.kind,
            "every_seconds": job.every_seconds,
            "next_run_at": job.next_run_at,
            "fires_at_local": _local_stamp(job.next_run_at, timezone),
            "local_now": _local_stamp(now, timezone),
            "once": job.once,
            "verbatim": job.verbatim,
        }
        if job.verbatim:
            payload["say"] = describe_saved_reminder(
                prompt=job.prompt,
                fires_at=job.next_run_at,
                now=now,
                timezone=timezone,
                name=job.name,
            )
            payload["note"] = (
                "The person is sent the say sentence ahead of your reply. "
                "Do not restate the reminder or its time; answer anything else they asked."
            )
        return json.dumps(payload, ensure_ascii=False)

    async def list_jobs(_arguments: dict[str, Any]) -> str:
        agent = active_agent()
        jobs = store.list_jobs(agent.name)
        payload = [
            {
                "id": job.id,
                "name": job.name,
                "kind": job.kind,
                "every_seconds": job.every_seconds,
                "next_run_at": job.next_run_at,
                "expires_at": job.expires_at,
                "once": job.once,
                "verbatim": job.verbatim,
                "relay": job.relay,
            }
            for job in jobs
        ]
        return json.dumps({"jobs": payload}, ensure_ascii=False)

    async def cancel(arguments: dict[str, Any]) -> str:
        agent = active_agent()
        name, job_id = parse_cancel(arguments)
        job = store.find_job(agent.name, name=name, job_id=job_id)
        if job is None:
            raise JobError("no matching job")
        store.cancel_job(job.id, at=_now())
        return json.dumps({"ok": True, "id": job.id, "name": job.name}, ensure_ascii=False)

    async def reschedule(arguments: dict[str, Any]) -> str:
        spec = parse_reschedule(arguments)
        agent = active_agent()
        job = store.find_job(agent.name, name=spec.name, job_id=spec.id)
        if job is None:
            raise JobError("no matching job")
        if job.ttl_seconds is not None and spec.every_seconds > job.ttl_seconds:
            raise JobError("every_seconds is longer than the watch TTL")
        now = _now()
        next_run_at = now + spec.every_seconds
        if not store.reschedule_job(
            job.id, every_seconds=spec.every_seconds, next_run_at=next_run_at
        ):
            raise JobError("no matching job")
        return json.dumps(
            {
                "ok": True,
                "id": job.id,
                "name": job.name,
                "every_seconds": spec.every_seconds,
                "next_run_at": next_run_at,
            },
            ensure_ascii=False,
        )

    return {
        JOB_CREATE: create,
        JOB_LIST: list_jobs,
        JOB_CANCEL: cancel,
        JOB_RESCHEDULE: reschedule,
    }


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
    now: float,
    timezone: str,
    audience: str = "you",
    name: str | None = None,
) -> str:
    """The sentence the person sees. The clock is the server's, not the model's."""
    require_representable(fires_at, timezone)
    require_representable(now, timezone)
    zone = ZoneInfo(timezone)
    when = datetime.fromtimestamp(fires_at, zone)
    current = datetime.fromtimestamp(now, zone)
    clock = when.strftime("%I:%M %p").lstrip("0")
    if when.date() == current.date():
        when_text = f"{clock} today"
    elif (when.date() - current.date()).days == 1:
        when_text = f"{clock} tomorrow"
    else:
        when_text = f"{when.strftime('%A, %B')} {when.day} at {clock}"
    sentence = f'I will text {audience} at {when_text}: "{prompt}"'
    if name:
        return f"{sentence}\nName: {name}"
    return sentence


def require_representable(moment: float, timezone: str) -> None:
    """Fail before a save when the instant cannot be shown in the household zone."""
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise JobError("the household timezone is not configured") from exc
    try:
        datetime.fromtimestamp(moment, zone)
    except (OSError, OverflowError, ValueError) as exc:
        raise JobError("that time is too far out to schedule") from exc


def _local_stamp(moment: float, timezone: str) -> str:
    require_representable(moment, timezone)
    return datetime.fromtimestamp(moment, ZoneInfo(timezone)).strftime("%Y-%m-%dT%H:%M:%S")


def _parse_delay(raw: object) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise JobError("in_seconds must be an integer")
    value = int(raw)
    if value != raw:
        raise JobError("in_seconds must be an integer")
    if value < MIN_EVERY_SECONDS:
        raise JobError("in_seconds must be at least 60")
    return value


def _parse_every(raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise JobError("every_seconds must be an integer")
    value = int(raw)
    if value != raw:
        raise JobError("every_seconds must be an integer")
    if value < MIN_EVERY_SECONDS:
        raise JobError("every_seconds must be at least 60")
    return value


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


def _parse_start_text(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise JobError("start_at must be an ISO-8601 date and time")
    text = raw.strip()
    if not text:
        raise JobError("start_at is empty")
    if "T" not in text:
        raise JobError("start_at must be an ISO-8601 date and time")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise JobError("start_at must be an ISO-8601 date and time") from exc
    if parsed.tzinfo is not None:
        raise JobError("start_at must not include a timezone suffix")
    return text


def _parse_id(raw: object) -> str:
    if not isinstance(raw, str) or not re.fullmatch(r"j[0-9a-f]{8}", raw.strip()):
        raise JobError("id is not a job id")
    return raw.strip()


def _now() -> float:
    return time.time()
