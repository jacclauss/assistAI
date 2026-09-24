"""Validate a filing batch or a draft before it is stored."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from assistai.errors import MailError

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ACTIONS = frozenset({"archive", "star", "unstar", "trash", "move"})
MAX_BATCH = 20
MAX_TEXT = 500
MAX_BODY = 8000


@dataclass(frozen=True)
class MailRef:
    id: str
    subject: str
    sender: str


@dataclass(frozen=True)
class FileBatch:
    action: str
    messages: tuple[MailRef, ...]
    label: str = ""

    def label_text(self) -> str:
        lines = [f"{self.action} {len(self.messages)}:"]
        if self.label:
            lines[0] = f"move {len(self.messages)} to {self.label}:"
        for item in self.messages:
            lines.append(f"- {item.sender}: {item.subject}")
        return "\n".join(lines)


@dataclass(frozen=True)
class DraftMessage:
    to: str
    subject: str
    body: str


def parse_message_id(value: object) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value.strip()):
        raise MailError("message id is not a Gmail id")
    return value.strip()


def parse_file(arguments: dict[str, Any]) -> FileBatch:
    action = arguments.get("action")
    if not isinstance(action, str) or action not in _ACTIONS:
        raise MailError("action must be archive, star, unstar, trash, or move")
    raw = arguments.get("messages")
    if not isinstance(raw, list) or not raw:
        raise MailError("messages must list the mail to change")
    if len(raw) > MAX_BATCH:
        raise MailError(f"a batch can change at most {MAX_BATCH} messages")
    messages = tuple(_ref(item) for item in raw)
    seen: set[str] = set()
    for item in messages:
        if item.id in seen:
            raise MailError("that batch lists the same message twice")
        seen.add(item.id)
    label = ""
    if action == "move":
        label = _line(arguments.get("label"), field="label", required=True)
    return FileBatch(action=action, messages=messages, label=label)


def parse_draft(arguments: dict[str, Any]) -> DraftMessage:
    return DraftMessage(
        to=_line(arguments.get("to", ""), field="to"),
        subject=_line(arguments.get("subject"), field="subject", required=True),
        body=_body(arguments.get("body")),
    )


def _ref(value: object) -> MailRef:
    if not isinstance(value, dict):
        raise MailError("each message needs an id, subject, and from")
    raw_id = parse_message_id(value.get("id"))
    return MailRef(
        id=raw_id,
        subject=_line(value.get("subject", ""), field="subject"),
        sender=_line(value.get("from", ""), field="from"),
    )


def _line(value: object, *, field: str, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise MailError(f"{field} must be text")
    text = " ".join(value.split())
    if required and not text:
        raise MailError(f"{field} is required")
    if len(text) > MAX_TEXT:
        raise MailError(f"{field} is too long")
    return text


def _body(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MailError("body is required")
    text = value.strip()
    if len(text) > MAX_BODY:
        raise MailError("body is too long")
    if "\x00" in text:
        raise MailError("body has characters that cannot be stored")
    return text
