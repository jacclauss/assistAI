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
    the provider, so a cut that lands on a tool result advances past it. Any
    other role is a valid start: an assistant tool-call keeps its results, and
    an assistant-only job report stands alone.
    """
    if len(messages) <= keep + 1:
        return
    cut = len(messages) - keep
    if cut < 1:
        cut = 1
    cut = _past_orphan_tools(messages, cut)
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
    cut = _past_orphan_tools(messages, cut)
    if cut > 1:
        del messages[1:cut]


def _past_orphan_tools(messages: list[Message], cut: int) -> int:
    """Advance a cut off a tool result whose assistant tool-call is being dropped.

    Only the tool results move the cut. Stepping further, to the next ``user``,
    would throw away every assistant-only job report that follows them.
    """
    while cut < len(messages) and messages[cut].role == "tool":
        cut += 1
    return cut


def trusted_context(messages: list[Message]) -> list[Message]:
    """History a job may send to the model. The stored window is unchanged.

    Untrusted text stays on disk so taint can age out, but a background run
    must not be steered by it or the digest launders the injection into a
    trusted assistant line. A tainted tool exchange is dropped as a whole so
    the provider never sees a dangling tool-call.
    """
    kept: list[Message] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.role == "assistant" and message.tool_calls:
            end = i + 1
            while end < len(messages) and messages[end].role == "tool":
                end += 1
            exchange = messages[i:end]
            if any(item.untrusted for item in exchange):
                i = end
                continue
            kept.extend(exchange)
            i = end
            continue
        if not message.untrusted:
            kept.append(message)
        i += 1
    return kept
