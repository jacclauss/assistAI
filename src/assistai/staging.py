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
from typing import Literal

from assistai.calendar.draft import CalendarDraft, parse_add
from assistai.errors import CalendarError
from assistai.inference.types import ToolCall

Decision = Literal["confirm", "reject"]

CONFIRM_RE = re.compile(r"^(yes|confirm)$", re.IGNORECASE)
REJECT_RE = re.compile(r"^(no|cancel)$", re.IGNORECASE)

_TAINT_NOTE = "Suggested while untrusted content was in context."
_CONFIRM_HINT_ONE = "Reply yes to run this, or no to discard."
_CONFIRM_HINT_MANY = "Reply yes to run these, or no to discard."


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


def format_proposal(calls: Sequence[ToolCall], *, tainted: bool) -> str:
    """Human-readable preview of the stored calls. This is what they confirm."""
    if len(calls) == 1 and calls[0].name == "relay":
        return _format_relay(calls[0], tainted=tainted)
    if len(calls) == 1 and calls[0].name == "job_create":
        return _format_job_create(calls[0], tainted=tainted)
    if len(calls) == 1 and calls[0].name == "job_reschedule":
        return _format_job_reschedule(calls[0], tainted=tainted)
    if len(calls) == 1 and calls[0].name == "job_cancel":
        return _format_job_cancel(calls[0], tainted=tainted)
    if len(calls) == 1 and calls[0].name == "calendar_add":
        return _format_calendar_add(calls[0], tainted=tainted)
    if len(calls) == 1 and calls[0].name == "mail_file":
        return _format_mail_file(calls[0], tainted=tainted)
    if len(calls) == 1 and calls[0].name == "mail_draft":
        return _format_mail_draft(calls[0], tainted=tainted)
    if len(calls) == 1 and calls[0].name == "shared_change":
        return _format_shared_change(calls[0], tainted=tainted)
    lines = [_heading(len(calls))]
    lines.extend(f"- {_proposal_line(call)}" for call in calls)
    lines.append("")
    lines.append(_CONFIRM_HINT_MANY if len(calls) != 1 else _CONFIRM_HINT_ONE)
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def format_executed(results: Sequence[tuple[ToolCall, str]]) -> str:
    """What ran, in the same order as the stored proposal."""
    if not results:
        return "Nothing to run."
    heading = "Ran 1 action:" if len(results) == 1 else f"Ran {len(results)} actions:"
    lines = [heading]
    lines.extend(f"- `{call.name}`: {body}" for call, body in results)
    return "\n".join(lines)


def _heading(count: int) -> str:
    if count == 1:
        return "I will run this when you confirm:\n"
    return f"I will run these {count} actions when you confirm:\n"


def _format_relay(call: ToolCall, *, tainted: bool) -> str:
    from assistai.relay import parse_body

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        body = parse_body(parsed) if isinstance(parsed, dict) else call.arguments
    except Exception:
        body = call.arguments
    lines = [
        "I will send this to the other phone when you confirm:\n",
        f"From you:\n{body}",
        "",
        "They will see it as a relay from you, not as their assistant.",
        "Reply yes to send this, no to discard, or say what to change.",
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _format_job_create(call: ToolCall, *, tainted: bool) -> str:
    from assistai.jobs import parse_create

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        spec = parse_create(parsed) if isinstance(parsed, dict) else None
    except Exception:
        spec = None
    if spec is None:
        return _generic_preview(call, tainted=tainted)
    kind = "schedule" if spec.kind == "schedule" else "watch"
    chatter = (
        "I will text you each time it runs, even if there is nothing to say."
        if spec.kind == "schedule"
        else "I will stay quiet unless it finds something, fails, or expires."
    )
    lines = [
        f"I will create this {kind} when you confirm:\n",
        f"Name: {spec.name}",
        f"Every: {spec.every_seconds} seconds",
        f"Check: {spec.prompt}",
    ]
    if spec.ttl_seconds is not None:
        lines.append(f"Ends after: {spec.ttl_seconds} seconds")
    lines.extend(["", chatter, _CONFIRM_HINT_ONE])
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _format_job_reschedule(call: ToolCall, *, tainted: bool) -> str:
    from assistai.jobs import parse_reschedule

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        spec = parse_reschedule(parsed) if isinstance(parsed, dict) else None
    except Exception:
        spec = None
    if spec is None:
        return _generic_preview(call, tainted=tainted)
    label = spec.name or spec.id or "this job"
    lines = [
        f"I will change {label} to every {spec.every_seconds} seconds when you confirm.\n",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _format_job_cancel(call: ToolCall, *, tainted: bool) -> str:
    from assistai.jobs import parse_cancel

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        name, job_id = parse_cancel(parsed) if isinstance(parsed, dict) else (None, None)
    except Exception:
        name, job_id = None, None
    if name is None and job_id is None:
        return _generic_preview(call, tainted=tainted)
    label = name or job_id or "this job"
    lines = [
        f"I will cancel {label} when you confirm.\n",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _generic_preview(call: ToolCall, *, tainted: bool) -> str:
    lines = [
        "I will run this when you confirm:\n",
        f"- {_proposal_line(call)}",
        "",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _proposal_line(call: ToolCall) -> str:
    if call.name == "job_create":
        line = _job_create_line(call)
        if line is not None:
            return line
    elif call.name == "job_cancel":
        line = _job_cancel_line(call)
        if line is not None:
            return line
    elif call.name == "job_reschedule":
        line = _job_reschedule_line(call)
        if line is not None:
            return line
    elif call.name == "calendar_add":
        line = _calendar_add_line(call)
        if line is not None:
            return line
    return f"`{call.name}` {_preview_args(call.arguments)}"


def _job_create_line(call: ToolCall) -> str | None:
    from assistai.jobs import parse_create

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        spec = parse_create(parsed) if isinstance(parsed, dict) else None
    except Exception:
        return None
    if spec is None:
        return None
    kind = "schedule" if spec.kind == "schedule" else "watch"
    line = f"Create {kind} {spec.name!r} every {spec.every_seconds} seconds: {spec.prompt}"
    if spec.ttl_seconds is not None:
        line += f" (ends after {spec.ttl_seconds} seconds)"
    return line


def _job_cancel_line(call: ToolCall) -> str | None:
    from assistai.jobs import parse_cancel

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        name, job_id = parse_cancel(parsed) if isinstance(parsed, dict) else (None, None)
    except Exception:
        return None
    if name is None and job_id is None:
        return None
    return f"Cancel {name or job_id}"


def _job_reschedule_line(call: ToolCall) -> str | None:
    from assistai.jobs import parse_reschedule

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        spec = parse_reschedule(parsed) if isinstance(parsed, dict) else None
    except Exception:
        return None
    if spec is None:
        return None
    label = spec.name or spec.id or "this job"
    return f"Change {label} to every {spec.every_seconds} seconds"


def _format_calendar_add(call: ToolCall, *, tainted: bool) -> str:
    draft = _calendar_draft(call)
    if draft is None:
        return _generic_preview(call, tainted=tainted)
    lines = [
        "I will add this to the shared calendar when you confirm:\n",
        draft.summary,
        draft.label(),
    ]
    if draft.location:
        lines.append(f"Location: {draft.location}")
    if draft.description:
        lines.append(draft.description)
    lines.extend(["", _CONFIRM_HINT_ONE])
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _calendar_add_line(call: ToolCall) -> str | None:
    draft = _calendar_draft(call)
    if draft is None:
        return None
    return f"Add {draft.summary!r} {draft.label()}"


def _calendar_draft(call: ToolCall) -> CalendarDraft | None:
    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        if not isinstance(parsed, dict):
            return None
        return parse_add(parsed)
    except (CalendarError, json.JSONDecodeError):
        return None


def _format_mail_file(call: ToolCall, *, tainted: bool) -> str:
    from assistai.errors import MailError
    from assistai.mail.actions import parse_file

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        batch = parse_file(parsed) if isinstance(parsed, dict) else None
    except (MailError, json.JSONDecodeError):
        batch = None
    if batch is None:
        return _generic_preview(call, tainted=tainted)
    lines = [
        "I will change this mail when you confirm:\n",
        batch.label_text(),
        "",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _format_shared_change(call: ToolCall, *, tainted: bool) -> str:
    from assistai.errors import SharedError
    from assistai.shared.records import parse_change

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        change = parse_change(parsed) if isinstance(parsed, dict) else None
    except (SharedError, json.JSONDecodeError):
        change = None
    if change is None:
        return _generic_preview(call, tainted=tainted)
    lines = [
        "I will change this shared list when you confirm:\n",
        change.label_text(),
        "",
        _CONFIRM_HINT_ONE,
    ]
    if tainted:
        lines.append(_TAINT_NOTE)
    return "\n".join(lines)


def _format_mail_draft(call: ToolCall, *, tainted: bool) -> str:
    from assistai.errors import MailError
    from assistai.mail.actions import parse_draft

    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        draft = parse_draft(parsed) if isinstance(parsed, dict) else None
    except (MailError, json.JSONDecodeError):
        draft = None
    if draft is None:
        return _generic_preview(call, tainted=tainted)
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
