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
from dataclasses import dataclass
from typing import Any

from assistai.agents import AgentSpec, Household
from assistai.errors import JobError, StoreError
from assistai.errors import RelayError as RelayError
from assistai.inference.types import ToolSpec
from assistai.signal.client import MAX_SEND_CHARS, SignalTransport
from assistai.signal.numbers import InvalidNumberError, normalize_e164

_DEFAULT_RELAY_NAME = "message to spouse"

RELAY_TOOL = "relay"

RELAY_SPEC = ToolSpec(
    name=RELAY_TOOL,
    description=(
        "Send a Signal message to the other household member, now or later. "
        "'Remind them in 2 minutes that X' is this tool: body is the sentence "
        "they should read, in_seconds is 120. Call it. Do not reply that you cannot. "
        "Draft the body. Do not ask them to dictate the wording. "
        "Nothing is sent until they reply yes. "
        "in_seconds is a delay the server counts. start_at is a named local time "
        "and needs get_time in this turn. "
        "Once they reply yes, that stored text goes out at that time. "
        "Omit both to send as soon as they say yes. "
        "once defaults to true. Set once false and every_seconds to repeat until they cancel. "
        "Attachments are not supported yet — ask them to paste a link instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "description": "Draft the other person will read, written by you.",
            },
            "in_seconds": {
                "type": "integer",
                "description": (
                    "Delay until the send, in seconds, measured by the server. "
                    "At least 60. Omit start_at when you set this."
                ),
            },
            "start_at": {
                "type": "string",
                "description": (
                    "Named send time, local ISO-8601 without a timezone suffix. "
                    "Requires get_time in this turn."
                ),
            },
            "once": {
                "type": "boolean",
                "description": "Send once, then stop. Defaults to true for a future message.",
            },
            "every_seconds": {
                "type": "integer",
                "description": "Gap between repeats after the first send. 86400 is daily.",
            },
            "name": {
                "type": "string",
                "description": (
                    "Name used to cancel a future message. Defaults to 'message to spouse'."
                ),
            },
        },
        "required": ["body"],
        "additionalProperties": False,
    },
)

InjectFn = Callable[[AgentSpec, str], Awaitable[None]]

_ACTIVE_AGENT: ContextVar[AgentSpec | None] = ContextVar("assistai_relay_agent", default=None)


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


@dataclass(frozen=True)
class RelayWhen:
    """A future send. Absent fields mean the relay goes out on yes."""

    name: str
    every_seconds: int
    once: bool
    in_seconds: int | None
    start_at: str | None


def parse_relay_when(arguments: dict[str, Any]) -> RelayWhen | None:
    """Timing for a later relay, or None when yes should send now.

    A start_at that is present but not text used to fall through to an
    immediate send. Yes would deliver now, which is the opposite of a
    named time.
    """
    from assistai.jobs import parse_create

    in_raw = arguments.get("in_seconds") if arguments.get("in_seconds") is not None else None
    if "start_at" in arguments and arguments.get("start_at") is not None:
        start_raw = arguments.get("start_at")
        if not isinstance(start_raw, str) or not start_raw.strip():
            raise RelayError("start_at must be an ISO-8601 date and time")
    else:
        start_raw = None
    if in_raw is None and start_raw is None:
        return None
    name = arguments.get("name") if arguments.get("name") is not None else _DEFAULT_RELAY_NAME
    once = True if arguments.get("once") is None else arguments.get("once")
    every = 86400 if arguments.get("every_seconds") is None else arguments.get("every_seconds")
    try:
        spec = parse_create(
            {
                "kind": "schedule",
                "name": name,
                "prompt": "relay",
                "every_seconds": every,
                "once": once,
                "verbatim": True,
                **({"in_seconds": in_raw} if in_raw is not None else {}),
                **(
                    {"start_at": start_raw}
                    if isinstance(start_raw, str) and start_raw.strip()
                    else {}
                ),
            }
        )
    except JobError as exc:
        raise RelayError(str(exc)) from exc
    return RelayWhen(
        name=spec.name,
        every_seconds=spec.every_seconds,
        once=spec.once,
        in_seconds=spec.in_seconds,
        start_at=spec.start_at,
    )


def when_phrase(when: RelayWhen) -> str:
    """Preview text. The clock is resolved again when they say yes."""
    if when.in_seconds is not None:
        return f"{_delay_phrase(when.in_seconds)} after you confirm"
    return f"at {when.start_at} local time"


def _delay_phrase(seconds: int) -> str:
    if seconds % 86400 == 0:
        count = seconds // 86400
        unit = "day" if count == 1 else "days"
        return f"{count} {unit}"
    if seconds % 3600 == 0:
        count = seconds // 3600
        unit = "hour" if count == 1 else "hours"
        return f"{count} {unit}"
    if seconds % 60 == 0:
        count = seconds // 60
        unit = "minute" if count == 1 else "minutes"
        return f"{count} {unit}"
    return f"{seconds} seconds"


async def deliver_stored_relay(
    *,
    household: Household,
    signal: SignalTransport,
    inject: InjectFn,
    sender: AgentSpec,
    body: str,
    on_sent: Callable[[], None] | None = None,
    send: bool = True,
    record: bool = True,
) -> None:
    """Send the stored bytes and record them for the recipient.

    A scheduled send marks the phone first, then records. The record step
    can be retried on its own. The sender is not told it went out until
    that record exists.
    """
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
    if send:
        outbound = format_outbound(sender.name, body)
        await signal.send(recipient.binding.peer, outbound)
        # The phone has the bytes. A scheduled retry must not send them again.
        if on_sent is not None:
            on_sent()
    if record:
        stored = format_record(sender.name, body)
        try:
            await inject(recipient, stored)
        except StoreError:
            raise RelayError("relay was sent but could not be recorded for the recipient") from None


def bind_relay(
    *,
    household: Household,
    signal: SignalTransport,
    inject: InjectFn,
    store: Any = None,
    timezone: str = "UTC",
) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """Handler closed over the live household and Signal client.

    ``inject`` writes the untrusted record into the recipient's conversation.
    The channel owns that write so it can take her history lock and keep the
    in-memory cache aligned with SQLite. A future relay stores the body and
    sends those bytes when it is due. Yes is the only confirmation.
    """

    async def relay(arguments: dict[str, Any]) -> str:
        sender = active_agent()
        body = parse_body(arguments, from_name=sender.name)
        when = parse_relay_when(arguments)
        if when is None:
            await deliver_stored_relay(
                household=household,
                signal=signal,
                inject=inject,
                sender=sender,
                body=body,
            )
            return json.dumps(
                {"ok": True, "to": _peer(household, sender), "from": sender.name},
                ensure_ascii=False,
            )
        if store is None:
            raise RelayError("a future relay cannot be stored")
        from assistai.jobs import (
            Job,
            current_time,
            describe_saved_reminder,
            new_job_id,
            require_representable,
            resolve_start_at,
        )

        now = current_time()
        if when.in_seconds is not None:
            first_run = now + when.in_seconds
        elif when.start_at is not None:
            try:
                first_run = resolve_start_at(when.start_at, now=now, timezone=timezone)
            except JobError as exc:
                raise RelayError(str(exc)) from exc
        else:
            raise RelayError("a future relay needs in_seconds or start_at")
        try:
            require_representable(first_run, timezone)
        except JobError as exc:
            raise RelayError(str(exc)) from exc
        job_name = _relay_job_name(
            store,
            sender.name,
            when.name,
            explicit=arguments.get("name") is not None and when.name != _DEFAULT_RELAY_NAME,
        )
        job = Job(
            id=new_job_id(),
            agent=sender.name,
            kind="schedule",
            name=job_name,
            prompt=body,
            every_seconds=when.every_seconds,
            ttl_seconds=None,
            created_at=now,
            expires_at=None,
            next_run_at=first_run,
            last_run_at=None,
            cancelled_at=None,
            once=when.once,
            verbatim=True,
            relay=True,
        )
        try:
            store.save_job(job)
        except StoreError as exc:
            raise RelayError("the future message could not be saved") from exc
        say = describe_saved_reminder(
            prompt=body,
            fires_at=first_run,
            now=now,
            timezone=timezone,
            audience="the other phone",
            name=job.name,
        )
        return json.dumps(
            {"ok": True, "scheduled": True, "id": job.id, "name": job.name, "say": say},
            ensure_ascii=False,
        )

    return relay


def _relay_job_name(store: Any, agent_name: str, requested: str, *, explicit: bool) -> str:
    """A second untagged message gets the next free default name.

    An explicit name that is already taken still fails. The default is what
    the model sends when the person did not pick one, and two of those must
    both be stored.
    """
    if store.find_job(agent_name, name=requested) is None:
        return requested
    if explicit:
        raise RelayError(f"a job named {requested!r} already exists")
    for index in range(2, 100):
        candidate = f"{requested} {index}"
        if store.find_job(agent_name, name=candidate) is None:
            return candidate
    raise RelayError(f"a job named {requested!r} already exists")


def _peer(household: Household, sender: AgentSpec) -> str:
    recipient = _other_dm(household, sender.name)
    if recipient is None:
        return ""
    return recipient.binding.peer


def _other_dm(household: Household, sender_name: str) -> AgentSpec | None:
    others = [
        agent
        for agent in household.agents
        if agent.binding.is_signal_dm and agent.name != sender_name
    ]
    if len(others) != 1:
        return None
    return others[0]
