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
    lines = [_heading(len(calls))]
    lines.extend(f"- `{call.name}` {_preview_args(call.arguments)}" for call in calls)
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


def _preview_args(raw: str) -> str:
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, dict):
        return raw
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
