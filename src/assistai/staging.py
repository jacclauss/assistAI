"""Staged actions: store the resolved call, wait, execute those bytes.

The model never gets a second chance to re-render between preview and
execution. Confirmation is a whole-message yes or no; the preview the person
sees is formatted from the stored calls, not from the model's prose.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from assistai.calendar.draft import CalendarDraft, parse_add
from assistai.clock import (
    Delay,
    HouseholdClock,
    LocalAt,
    resolve_at,
    say_at,
    say_duration,
    say_every,
    say_repeat,
    say_when,
)
from assistai.errors import CalendarError, JobError
from assistai.inference.types import ToolCall

if TYPE_CHECKING:
    from assistai.jobs import Job, JobReschedule

Decision = Literal["confirm", "reject"]

CONFIRM_RE = re.compile(r"^(yes|confirm)$", re.IGNORECASE)
REJECT_RE = re.compile(r"^(no|cancel)$", re.IGNORECASE)

_TAINT_NOTE = "Suggested while untrusted content was in context."
_CONFIRM_HINT_ONE = "Reply yes to run this, or no to discard."
_CONFIRM_HINT_MANY = "Reply yes to run these, or no to discard."


class JobFinder(Protocol):
    """The acting person's job by name or id, so a reschedule preview can name its next run."""

    def __call__(self, *, name: str | None, job_id: str | None) -> Job | None: ...


@dataclass(frozen=True)
class Proposal:
    """One live proposal per agent. A newer one replaces an older one."""

    agent: str
    calls: tuple[ToolCall, ...]
    tainted: bool
    created_at: float
    expires_at: float


def parse_decision(text: str) -> Decision | None:
    """Return confirm/reject only when the whole message is a decision."""
    stripped = text.strip()
    if CONFIRM_RE.fullmatch(stripped):
        return "confirm"
    if REJECT_RE.fullmatch(stripped):
        return "reject"
    return None


def format_proposal(
    calls: Sequence[ToolCall],
    *,
    tainted: bool,
    clock: HouseholdClock,
    jobs: JobFinder | None = None,
) -> str:
    """Human-readable preview of the stored calls. This is what they confirm.

    Every time in it is the server's words on the household clock, so the
    day a yes runs is the day the person read. ``jobs`` finds a job a
    reschedule moves; without it a preview names only what the call says.
    """
    words = _Words(clock=clock, jobs=jobs)
    if len(calls) == 1 and calls[0].name == "relay":
        return _format_relay(calls[0], tainted=tainted, words=words)
    if len(calls) == 1 and calls[0].name == "job_create":
        return _format_job_create(calls[0], tainted=tainted, words=words)
    if len(calls) == 1 and calls[0].name == "job_reschedule":
        return _format_job_reschedule(calls[0], tainted=tainted, words=words)
    if len(calls) == 1 and calls[0].name == "job_cancel":
        return _format_job_cancel(calls[0], tainted=tainted, words=words)
    if len(calls) == 1 and calls[0].name == "calendar_add":
        return _format_calendar_add(calls[0], tainted=tainted, words=words)
    if len(calls) == 1 and calls[0].name == "mail_file":
        return _format_mail_file(calls[0], tainted=tainted, words=words)
    if len(calls) == 1 and calls[0].name == "mail_draft":
        return _format_mail_draft(calls[0], tainted=tainted, words=words)
    if len(calls) == 1 and calls[0].name == "shared_change":
        return _format_shared_change(calls[0], tainted=tainted, words=words)
    lines = [_heading(len(calls))]
    lines.extend(f"- {_proposal_line(call, words=words)}" for call in calls)
    lines.append("")
    lines.append(_CONFIRM_HINT_MANY if len(calls) != 1 else _CONFIRM_HINT_ONE)
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def format_executed(results: Sequence[tuple[ToolCall, str]]) -> str:
    """What ran, in the same order as the stored proposal."""
    if not results:
        return "Nothing to run."
    said = [_saved_line(body) for _, body in results]
    if all(line is not None for line in said):
        return "\n".join(line for line in said if line is not None)
    heading = "Ran 1 action:" if len(results) == 1 else f"Ran {len(results)} actions:"
    lines = [heading]
    lines.extend(f"- `{call.name}`: {body}" for call, body in results)
    return "\n".join(lines)


def _saved_line(body: str) -> str | None:
    try:
        parsed: object = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    say = parsed.get("say")
    if isinstance(say, str) and say:
        return say
    return None


@dataclass(frozen=True)
class _Words:
    """What a preview needs to put a stored time into words."""

    clock: HouseholdClock
    jobs: JobFinder | None = None

    def at(self, at: LocalAt) -> str:
        """A staged time on the household clock. One that has passed reads as named."""
        try:
            return say_when(resolve_at(at, clock=self.clock), clock=self.clock)
        except JobError:
            return say_at(at)


def _heading(count: int) -> str:
    if count == 1:
        return "I will run this when you confirm:\n"
    return f"I will run these {count} actions when you confirm:\n"


def _format_relay(call: ToolCall, *, tainted: bool, words: _Words) -> str:
    from assistai.relay import parse_body, parse_relay_when

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        body = parse_body(parsed, from_name="you") if isinstance(parsed, dict) else call.arguments
        when = parse_relay_when(parsed) if isinstance(parsed, dict) else None
    except Exception:
        body = call.arguments
        when = None
    if when is None:
        lead = "I will send this to the other phone when you confirm:\n"
        later = ""
    else:
        from assistai.relay import when_phrase

        # Only a named time needs "when you confirm"; the rest count from the yes.
        confirm = " when you confirm" if isinstance(when.first, LocalAt) else ""
        phrase = when_phrase(when, clock=words.clock)
        lead = f"I will send this to the other phone {phrase}{confirm}:\n"
        repeat = say_every(
            every_days=when.every_days,
            every_minutes=when.every_minutes,
            at=when.first if isinstance(when.first, LocalAt) else None,
        )
        later = "" if when.once else f"It repeats {repeat} until you cancel it.\n"
    lines = [
        lead,
        f"From you:\n{body}",
        "",
        "They will see it as a relay from you, not as their assistant.",
        later,
        "Reply yes to send this, no to discard, or say what to change.",
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(line for line in lines if line != "")


def _format_job_create(call: ToolCall, *, tainted: bool, words: _Words) -> str:
    from assistai.jobs import parse_create

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        spec = parse_create(parsed) if isinstance(parsed, dict) else None
    except Exception:
        spec = None
    if spec is None:
        return _generic_preview(call, tainted=tainted, words=words)
    kind = "schedule" if spec.kind == "schedule" else "watch"
    if spec.once:
        chatter = "I will text you once, then this job ends."
    elif spec.kind == "schedule":
        chatter = "I will text you each time it runs, even if there is nothing to say."
    else:
        chatter = "I will stay quiet unless it finds something, fails, or expires."
    lines = [
        f"I will create this {kind} when you confirm:\n",
        f"Name: {spec.name}",
        f"Check: {spec.prompt}",
        f"First run: {_first_run_text(spec.first, every_minutes=spec.every_minutes, words=words)}",
        f"Repeats: {_repeat_text(spec.every_days, spec.every_minutes, spec.first)}",
    ]
    if spec.verbatim:
        lines.append("The text above is what you will receive.")
    if spec.ttl_seconds is not None:
        lines.append(f"Ends after: {spec.ttl_seconds} seconds")
    lines.extend(["", chatter, _CONFIRM_HINT_ONE])
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _first_run_text(
    first: LocalAt | Delay | None, *, every_minutes: int | None, words: _Words
) -> str:
    """When the first run is, from the stored call. A delay counts from the yes, not the preview."""
    if isinstance(first, LocalAt):
        return words.at(first)
    if isinstance(first, Delay):
        return f"{say_duration(first.seconds)} after you reply yes"
    if every_minutes is not None:
        return f"{say_duration(every_minutes * 60)} after you reply yes"
    return "when you reply yes"


def _repeat_text(
    every_days: int | None, every_minutes: int | None, first: LocalAt | Delay | None
) -> str:
    return say_every(
        every_days=every_days,
        every_minutes=every_minutes,
        at=first if isinstance(first, LocalAt) else None,
    )


def _format_job_reschedule(call: ToolCall, *, tainted: bool, words: _Words) -> str:
    from assistai.jobs import parse_reschedule

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        spec = parse_reschedule(parsed) if isinstance(parsed, dict) else None
    except Exception:
        spec = None
    if spec is None:
        return _generic_preview(call, tainted=tainted, words=words)
    label = spec.name or spec.id or "this job"
    lines = [f"I will change {label} when you confirm:\n", *_reschedule_parts(spec, words=words)]
    lines.extend(["", _CONFIRM_HINT_ONE])
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _reschedule_parts(spec: JobReschedule, *, words: _Words) -> list[str]:
    """The next run and repeat the yes stores, worked out the way the handler will.

    A change to the repeat alone keeps the next run, so the preview names
    that run from the job itself. Without the job, it says what the call does.
    """
    from assistai.jobs import rescheduled

    job = words.jobs(name=spec.name, job_id=spec.id) if words.jobs is not None else None
    if job is not None:
        try:
            schedule, _gap = rescheduled(job, spec, clock=words.clock)
        except JobError:
            pass
        else:
            next_run = say_when(schedule.anchor_at, clock=words.clock)
            if not schedule.repeats:
                return [f"Next run: {next_run}"]
            return [f"Next run: {next_run}, then {say_repeat(schedule, clock=words.clock)}"]
    parts: list[str] = []
    if spec.at is not None:
        parts.append(f"Next run: {words.at(spec.at)}")
    if spec.every_days is not None or spec.every_minutes is not None:
        repeat = say_every(every_days=spec.every_days, every_minutes=spec.every_minutes, at=spec.at)
        parts.append(f"Repeats: {repeat}")
    return parts


def _format_job_cancel(call: ToolCall, *, tainted: bool, words: _Words) -> str:
    from assistai.jobs import parse_cancel

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        name, job_id = parse_cancel(parsed) if isinstance(parsed, dict) else (None, None)
    except Exception:
        name, job_id = None, None
    if name is None and job_id is None:
        return _generic_preview(call, tainted=tainted, words=words)
    label = name or job_id or "this job"
    lines = [
        f"I will cancel {label} when you confirm.\n",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _generic_preview(call: ToolCall, *, tainted: bool, words: _Words) -> str:
    lines = [
        "I will run this when you confirm:\n",
        f"- {_proposal_line(call, words=words)}",
        "",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _proposal_line(call: ToolCall, *, words: _Words) -> str:
    if call.name == "job_create":
        line = _job_create_line(call, words=words)
        if line is not None:
            return line
    elif call.name == "job_cancel":
        line = _job_cancel_line(call, words=words)
        if line is not None:
            return line
    elif call.name == "job_reschedule":
        line = _job_reschedule_line(call, words=words)
        if line is not None:
            return line
    elif call.name == "calendar_add":
        line = _calendar_add_line(call, words=words)
        if line is not None:
            return line
    elif call.name == "relay":
        line = _relay_line(call, words=words)
        if line is not None:
            return line
    return f"`{call.name}` {_preview_args(call.arguments)}"


def _relay_line(call: ToolCall, *, words: _Words) -> str | None:
    """A relay in a bundle names when it goes out, the same as a relay on its own."""
    from assistai.relay import parse_body, parse_relay_when, when_phrase

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        if not isinstance(parsed, dict):
            return None
        body = parse_body(parsed, from_name="you")
        when = parse_relay_when(parsed)
    except Exception:
        return None
    if when is None:
        return f"Send to the other phone when you confirm: {body}"
    line = f"Send to the other phone {when_phrase(when, clock=words.clock)}"
    if not when.once:
        repeat = say_every(
            every_days=when.every_days,
            every_minutes=when.every_minutes,
            at=when.first if isinstance(when.first, LocalAt) else None,
        )
        line += f", repeats {repeat}"
    return f"{line}: {body}"


def _job_create_line(call: ToolCall, *, words: _Words) -> str | None:
    from assistai.jobs import parse_create

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        spec = parse_create(parsed) if isinstance(parsed, dict) else None
    except Exception:
        return None
    if spec is None:
        return None
    kind = "schedule" if spec.kind == "schedule" else "watch"
    first = _first_run_text(spec.first, every_minutes=spec.every_minutes, words=words)
    repeat = _repeat_text(spec.every_days, spec.every_minutes, spec.first)
    line = f"Create {kind} {spec.name!r}, first run {first}, repeats {repeat}: {spec.prompt}"
    if spec.ttl_seconds is not None:
        line += f" (ends after {spec.ttl_seconds} seconds)"
    return line


def _job_cancel_line(call: ToolCall, *, words: _Words) -> str | None:
    from assistai.jobs import parse_cancel

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        name, job_id = parse_cancel(parsed) if isinstance(parsed, dict) else (None, None)
    except Exception:
        return None
    if name is None and job_id is None:
        return None
    return f"Cancel {name or job_id}"


def _job_reschedule_line(call: ToolCall, *, words: _Words) -> str | None:
    from assistai.jobs import parse_reschedule

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        spec = parse_reschedule(parsed) if isinstance(parsed, dict) else None
    except Exception:
        return None
    if spec is None:
        return None
    label = spec.name or spec.id or "this job"
    return f"Change {label}: " + "; ".join(_reschedule_parts(spec, words=words))


def _format_calendar_add(call: ToolCall, *, tainted: bool, words: _Words) -> str:
    draft = _calendar_draft(call)
    if draft is None:
        return _generic_preview(call, tainted=tainted, words=words)
    lines = [
        "I will add this to the shared calendar when you confirm:\n",
        draft.summary,
        draft.label(words.clock),
    ]
    if draft.location:
        lines.append(f"Location: {draft.location}")
    if draft.description:
        lines.append(draft.description)
    lines.extend(["", _CONFIRM_HINT_ONE])
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _calendar_add_line(call: ToolCall, *, words: _Words) -> str | None:
    draft = _calendar_draft(call)
    if draft is None:
        return None
    return f"Add {draft.summary!r} {draft.label(words.clock)}"


def _calendar_draft(call: ToolCall) -> CalendarDraft | None:
    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        if not isinstance(parsed, dict):
            return None
        return parse_add(parsed)
    except (CalendarError, json.JSONDecodeError):
        return None


def _format_mail_file(call: ToolCall, *, tainted: bool, words: _Words) -> str:
    from assistai.errors import MailError
    from assistai.mail.actions import parse_file

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        batch = parse_file(parsed) if isinstance(parsed, dict) else None
    except (MailError, json.JSONDecodeError):
        batch = None
    if batch is None:
        return _generic_preview(call, tainted=tainted, words=words)
    lines = [
        "I will change this mail when you confirm:\n",
        batch.label_text(),
        "",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _format_shared_change(call: ToolCall, *, tainted: bool, words: _Words) -> str:
    from assistai.errors import SharedError
    from assistai.shared.records import parse_change

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        change = parse_change(parsed) if isinstance(parsed, dict) else None
    except (SharedError, json.JSONDecodeError):
        change = None
    if change is None:
        return _generic_preview(call, tainted=tainted, words=words)
    if change.share:
        heading = "I will share this private list when you confirm:\n"
    elif change.private:
        heading = "I will change this private list when you confirm:\n"
    else:
        heading = "I will change this shared list when you confirm:\n"
    lines = [
        heading,
        change.label_text(),
        "",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _format_mail_draft(call: ToolCall, *, tainted: bool, words: _Words) -> str:
    from assistai.errors import MailError
    from assistai.mail.actions import parse_draft

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        draft = parse_draft(parsed) if isinstance(parsed, dict) else None
    except (MailError, json.JSONDecodeError):
        draft = None
    if draft is None:
        return _generic_preview(call, tainted=tainted, words=words)
    lines = [
        "I will save this Gmail draft when you confirm. It will not be sent.\n",
        f"To: {draft.to or '(none)'}",
        f"Subject: {draft.subject}",
        "",
        draft.body,
        "",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _preview_args(raw: str) -> str:
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, dict):
        return raw
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
