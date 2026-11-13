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
from assistai.clock import (
    AT_SCHEMA,
    DELAY_SCHEMA,
    EVERY_DAYS_SCHEMA,
    EVERY_MINUTES_SCHEMA,
    Delay,
    LocalAt,
    say_at,
    say_duration,
)
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
        "they should read, delay is minutes 2. Call it. Do not reply that you cannot. "
        "Draft the body. Do not ask them to dictate the wording. "
        "Nothing is sent until they reply yes. "
        "delay is a duration the person said, counted by the server from the yes. "
        "at is a clock time they named, in their words ('tomorrow' is day, a "
        "calendar date is date, a bare hour is neither), and needs get_time in this turn. "
        "Once they reply yes, that stored text goes out at that time. "
        "Omit both to send as soon as they say yes. "
        "once defaults to true. To repeat until they cancel, set once false and "
        "every_days with at for a daily or weekly message, or every_minutes. "
        "If they want it daily and did not name the time of day, ask what time first. "
        "Attachments are not supported yet — ask them to paste a link instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "description": "Draft the other person will read, written by you.",
            },
            "at": AT_SCHEMA,
            "delay": DELAY_SCHEMA,
            "once": {
                "type": "boolean",
                "description": "Send once, then stop. Defaults to true for a future message.",
            },
            "every_days": EVERY_DAYS_SCHEMA,
            "every_minutes": EVERY_MINUTES_SCHEMA,
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
_ACTIVE_TAINT: ContextVar[bool] = ContextVar("assistai_relay_tainted", default=False)


@contextmanager
def using_agent(agent: AgentSpec, *, tainted: bool = False) -> Iterator[None]:
    """Bind the confirming agent, and whether its conversation is tainted, for one handler call.

    A handler that stores text for later reads the taint here. The model's
    arguments cannot clear it.
    """
    token = activate_agent(agent)
    taint_token = _ACTIVE_TAINT.set(tainted)
    try:
        yield
    finally:
        _ACTIVE_TAINT.reset(taint_token)
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


def active_taint() -> bool:
    """True when the conversation running this handler holds untrusted text."""
    return _ACTIVE_TAINT.get()


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
    """A future send. ``first`` is the named time or the delay the person said."""

    name: str
    once: bool
    first: LocalAt | Delay | None
    every_days: int | None = None
    every_minutes: int | None = None


# Any of these makes a relay a later send, so it goes through the job parser.
# An old in_seconds or start_at must be refused there, not read as "send now".
_WHEN_FIELDS = (
    "at",
    "delay",
    "every_days",
    "every_minutes",
    "every_seconds",
    "in_seconds",
    "start_at",
)


def parse_relay_when(arguments: dict[str, Any]) -> RelayWhen | None:
    """Timing for a later relay, or None when yes should send now.

    The job parser's rules hold here too: a daily relay with no named hour
    is refused so the person is asked what time, and the old time arguments
    are refused instead of falling through to an immediate send, which is
    the opposite of a named time.
    """
    from assistai.jobs import parse_create

    if all(arguments.get(field) is None for field in _WHEN_FIELDS):
        return None
    name = arguments.get("name") if arguments.get("name") is not None else _DEFAULT_RELAY_NAME
    once = True if arguments.get("once") is None else arguments.get("once")
    try:
        spec = parse_create(
            {
                **{field: arguments[field] for field in _WHEN_FIELDS if field in arguments},
                "kind": "schedule",
                "name": name,
                "prompt": "relay",
                "once": once,
                "verbatim": True,
            }
        )
    except JobError as exc:
        raise RelayError(str(exc)) from exc
    return RelayWhen(
        name=spec.name,
        once=spec.once,
        first=spec.first,
        every_days=spec.every_days,
        every_minutes=spec.every_minutes,
    )


def when_phrase(when: RelayWhen) -> str:
    """Preview text. A staged at already carries its date; a delay counts from the yes."""
    if isinstance(when.first, Delay):
        return f"{say_duration(when.first.seconds)} after you reply yes"
    if isinstance(when.first, LocalAt):
        said = say_at(when.first)
        return f"on {said}" if when.first.date is not None else said
    return "as soon as you reply yes"


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
            describe_saved_reminder,
            first_run_at,
            household_clock,
            new_job_id,
            require_representable,
            schedule_for,
        )

        try:
            clock = household_clock(timezone)
            now = clock.now()
            first_run = first_run_at(when.first, clock=clock, every_minutes=when.every_minutes)
            require_representable(first_run, timezone)
            schedule = schedule_for(
                zone=clock.zone,
                first_run=first_run,
                once=when.once,
                first=when.first,
                every_days=when.every_days,
                every_minutes=when.every_minutes,
            )
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
            every_seconds=_every_seconds(when),
            ttl_seconds=None,
            created_at=now,
            expires_at=None,
            next_run_at=first_run,
            last_run_at=None,
            cancelled_at=None,
            once=when.once,
            verbatim=True,
            relay=True,
            tainted=active_taint(),
            zone=schedule.zone.key,
            anchor_at=schedule.anchor_at,
            every_days=schedule.every_days,
            at_local=schedule.at_local,
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


def _every_seconds(when: RelayWhen) -> int:
    """The gap written beside the schedule for an older build; a day for a one-time send."""
    if when.every_days is not None:
        return when.every_days * 86400
    if when.every_minutes is not None:
        return when.every_minutes * 60
    return 86400


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
