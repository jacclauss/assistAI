"""Conversation helpers shared by the REPL and the Signal channel.

Kept out of ``chat`` so the Signal path does not import the terminal REPL.
"""

from __future__ import annotations

from assistai.inference.types import Message

SYSTEM_PROMPT = (
    "You are AssistAI, a concise household assistant. "
    "Use tools when they help answer the question. "
    "Do not invent tool results."
)


def last_assistant_text(messages: list[Message]) -> str:
    """The most recent assistant reply that was not a tool request."""
    for message in reversed(messages):
        if message.role == "assistant" and not message.tool_calls:
            return message.content or ""
    return ""


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
