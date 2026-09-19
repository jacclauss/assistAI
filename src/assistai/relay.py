"""Cross-person Signal relay. Staging, not a free message:other_peer sink.

The stored call is the body the sender confirmed. The handler sends those
bytes to the recipient's number, labelled so it is obviously a relay, then
appends an untrusted record to the recipient's history. Confirmation
authorized delivery to the phone, not privileged sinks on her side.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from assistai.agents import AgentSpec, Household
from assistai.errors import AssistAIError, StoreError
from assistai.inference.types import ToolSpec
from assistai.signal.client import MAX_SEND_CHARS, SignalTransport
from assistai.signal.numbers import InvalidNumberError, normalize_e164

RELAY_TOOL = "relay"

RELAY_SPEC = ToolSpec(
    name=RELAY_TOOL,
    description=(
        "Propose a verbatim Signal message to the other household member. "
        "The body is what they will receive after the sender confirms. "
        "Do not invent a body; use the sender's words. Attachments are not "
        "supported yet — tell the sender to paste a link instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "description": "Exact text to send. No extra greeting or rewrite.",
            }
        },
        "required": ["body"],
        "additionalProperties": False,
    },
)

InjectFn = Callable[[AgentSpec, str], Awaitable[None]]

_ACTIVE_AGENT: ContextVar[AgentSpec | None] = ContextVar("assistai_relay_agent", default=None)


class RelayError(AssistAIError):
    """The stored relay could not be delivered."""


@contextmanager
def using_agent(agent: AgentSpec) -> Iterator[None]:
    """Bind the confirming agent for the duration of a handler call."""
    token = activate_agent(agent)
    try:
        yield
    finally:
        reset_agent(token)


def activate_agent(agent: AgentSpec) -> Token[AgentSpec | None]:
    return _ACTIVE_AGENT.set(agent)


def reset_agent(token: Token[AgentSpec | None]) -> None:
    _ACTIVE_AGENT.reset(token)


def active_agent() -> AgentSpec:
    agent = _ACTIVE_AGENT.get()
    if agent is None:
        raise RelayError("relay has no active agent")
    return agent


def format_outbound(from_name: str, body: str) -> str:
    """What the recipient's phone shows. Not her assistant speaking."""
    return f"From {from_name.title()}:\n{body}"


def format_record(from_name: str, body: str) -> str:
    """What her assistant sees later. Untrusted: it may carry a page or mail."""
    return json.dumps(
        {"kind": "relay", "from": from_name, "body": body, "attachments": []},
        ensure_ascii=False,
        sort_keys=True,
    )


def parse_body(arguments: dict[str, Any], *, from_name: str | None = None) -> str:
    raw = arguments.get("body")
    if not isinstance(raw, str):
        raise RelayError("relay body must be a string")
    body = raw.strip()
    if not body:
        raise RelayError("relay body is empty")
    labelled = format_outbound(from_name or "x", body)
    if len(labelled) > MAX_SEND_CHARS:
        raise RelayError(
            f"relay body is too long ({len(labelled)} characters, limit {MAX_SEND_CHARS})"
        )
    return body


def bind_relay(
    *,
    household: Household,
    signal: SignalTransport,
    inject: InjectFn,
) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """Handler closed over the live household and Signal client.

    ``inject`` writes the untrusted record into the recipient's conversation.
    The channel owns that write so it can take her history lock and keep the
    in-memory cache aligned with SQLite.
    """

    async def relay(arguments: dict[str, Any]) -> str:
        sender = active_agent()
        if household.agent_for_signal_dm(sender.binding.peer) is None:
            raise RelayError("sender has no agent")
        recipient = _other_dm(household, sender.name)
        if recipient is None:
            raise RelayError("no other household member to relay to")
        try:
            if normalize_e164(sender.binding.peer) == recipient.binding.peer:
                raise RelayError("cannot relay to yourself")
        except InvalidNumberError as exc:
            raise RelayError("recipient number is invalid") from exc
        body = parse_body(arguments, from_name=sender.name)
        outbound = format_outbound(sender.name, body)
        await signal.send(recipient.binding.peer, outbound)
        record = format_record(sender.name, body)
        try:
            await inject(recipient, record)
        except StoreError:
            # The phone already has the bytes. Do not pretend the inject
            # succeeded; the sender can retry knowing delivery happened.
            raise RelayError("relay was sent but could not be recorded for the recipient") from None
        return json.dumps(
            {"ok": True, "to": recipient.binding.peer, "from": sender.name},
            ensure_ascii=False,
        )

    return relay


def _other_dm(household: Household, sender_name: str) -> AgentSpec | None:
    others = [
        agent
        for agent in household.agents
        if agent.binding.is_signal_dm and agent.name != sender_name
    ]
    if len(others) != 1:
        return None
    return others[0]
