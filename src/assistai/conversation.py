"""Conversation helpers shared by the REPL and the Signal channel.

Kept out of ``chat`` so the Signal path does not import the terminal REPL.
"""

from __future__ import annotations

import time

from assistai.inference.types import Message


def last_assistant_text(messages: list[Message]) -> str:
    """The most recent assistant reply that was not a tool request."""
    for message in reversed(messages):
        if message.role == "assistant" and not message.tool_calls:
            return message.content or ""
    return ""


def apply_window(
    messages: list[Message],
    *,
    keep: int,
    max_age_seconds: float,
    now: float | None = None,
) -> None:
    """Bound history by age, then by count. Never splits a tool exchange."""
    _trim_older_than(messages, (now if now is not None else time.time()) - max_age_seconds)
    safe_trim(messages, keep)


def safe_trim(messages: list[Message], keep: int) -> None:
    """Drop the oldest turns, never splitting an assistant/tool exchange.

    A ``tool`` message whose assistant tool-call was trimmed away is rejected by
    the provider, so the cut advances to the next ``user`` message, which is
    always a clean turn boundary.
    """
    if len(messages) <= keep + 1:
        return
    cut = len(messages) - keep
    while cut < len(messages) and messages[cut].role != "user":
        cut += 1
    del messages[1:cut]


def _trim_older_than(messages: list[Message], cutoff: float) -> None:
    """Drop leading non-system messages older than ``cutoff``.

    A message with no timestamp is treated as recent so in-memory tests and
    the current turn are not expired by accident.
    """
    if len(messages) <= 1:
        return
    cut = 1
    while cut < len(messages):
        stamp = messages[cut].created_at
        if stamp is None or stamp >= cutoff:
            break
        cut += 1
    else:
        del messages[1:]
        return
    while cut < len(messages) and messages[cut].role != "user":
        cut += 1
    if cut > 1:
        del messages[1:cut]
