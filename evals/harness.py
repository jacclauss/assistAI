"""Golden behaviour cases against a real model, with every tool faked.

The model, system prompt, catalog, and broker are the production ones. Tool
handlers return the case's canned fixtures, so a case can say "the inbox has
7 unread" or "this page carries an injection" and grade what the model did
with it. Nothing reaches Gmail, iCloud, the web, or a phone.

A case grades behaviour, not wording: which tools were called, which were
staged, the arguments of the first call of a tool, and a few substrings the
reply must or must not contain.

A case with ``now`` and ``timezone`` runs on that stopped household clock:
``get_time``, the prompt's local time, and the date a staged ``at`` is fixed
to all read it, so a time case grades the same on any day it is run.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from assistai.agents import AgentSpec, Binding, BrokerPolicy, load_household, system_prompt_for
from assistai.broker import Scope, ToolBroker, ToolCatalog, builtin_catalog
from assistai.clock import HouseholdClock, wall_time
from assistai.config import Settings
from assistai.conversation import last_assistant_text
from assistai.errors import AssistAIError, JobError
from assistai.inference.client import DeltaCallback, FireworksClient
from assistai.inference.loop import run_turn
from assistai.inference.tools import ToolHandler, bind_clock
from assistai.inference.types import Completion, Message, Role, ToolSpec
from assistai.jobs import DAY_SECONDS, JOB_LIST, Job, job_list_result, schedule_from_seconds
from assistai.manifest import ModelPin

CASES_DIR = Path(__file__).resolve().parent / "cases"
_DEFAULT_RESULT = json.dumps({"ok": True})
_HISTORY_ROLES: frozenset[Role] = frozenset({"user", "assistant"})
_AGENT = "jacob"
_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")
_ARGUMENT_KEYS = frozenset({"equals", "absent", "any_of"})
_MISSING = object()

Scalar = str | int | float | bool


class CaseError(ValueError):
    """A case file that cannot be graded as written."""


@dataclass(frozen=True)
class ArgumentExpect:
    """What the model's own arguments to one tool must hold, by dotted path.

    The model's call is graded, not the copy staging stores, because the
    question is whether it copied the person's words: "tomorrow" is
    ``at.day``, and the staged copy has already turned that into a date.
    """

    equals: Mapping[str, Scalar] = field(default_factory=dict)
    absent: tuple[str, ...] = ()
    any_of: tuple[Mapping[str, Scalar], ...] = ()


@dataclass(frozen=True)
class Expect:
    calls: tuple[str, ...] = ()
    no_calls: tuple[str, ...] = ()
    staged: tuple[str, ...] = ()
    no_staged: tuple[str, ...] = ()
    reply_contains: tuple[str, ...] = ()
    reply_excludes: tuple[str, ...] = ()
    max_tool_calls: int | None = None
    arguments: Mapping[str, ArgumentExpect] = field(default_factory=dict)


@dataclass(frozen=True)
class Case:
    id: str
    user: str
    tools: tuple[str, ...]
    web_access: bool
    expect: Expect
    fixtures: dict[str, str] = field(default_factory=dict)
    history: tuple[Message, ...] = ()
    tags: tuple[str, ...] = ()
    pass_rate: float = 1.0
    timezone: str = ""
    now: float | None = None
    jobs: tuple[Job, ...] = ()

    def clock(self) -> HouseholdClock | None:
        """The household clock stopped at ``now``, or None for a case without one."""
        if self.now is None:
            return None
        instant = self.now
        return HouseholdClock(zone=ZoneInfo(self.timezone), now=lambda: instant)


@dataclass(frozen=True)
class Trial:
    passed: bool
    failures: tuple[str, ...]
    calls: tuple[str, ...]
    staged: tuple[str, ...]
    reply: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    arguments: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseResult:
    case: Case
    trials: tuple[Trial, ...]

    @property
    def passes(self) -> int:
        return sum(1 for trial in self.trials if trial.passed)

    @property
    def passed(self) -> bool:
        return self.passes >= math.ceil(len(self.trials) * self.case.pass_rate)


class _MeteredClient(FireworksClient):
    """Counts tokens per trial. The loop logs usage but does not return it."""

    prompt_tokens = 0
    completion_tokens = 0

    async def complete(
        self,
        pin: ModelPin,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        on_delta: DeltaCallback | None = None,
    ) -> Completion:
        completion = await super().complete(pin, messages, tools, on_delta=on_delta)
        if completion.usage is not None:
            self.prompt_tokens += completion.usage.prompt_tokens
            self.completion_tokens += completion.usage.completion_tokens
        return completion


def load_cases(
    directory: Path = CASES_DIR, *, roster: Path, catalog: ToolCatalog | None = None
) -> list[Case]:
    """Parse every case file. ``roster`` supplies the default tool list."""
    known = (catalog or builtin_catalog()).names()
    household = load_household(roster, known_tools=known)
    default = household.agents[0]
    cases = [
        _parse_case(path, known=known, default=default) for path in sorted(directory.glob("*.toml"))
    ]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise CaseError(f"duplicate case id {case.id}")
        seen.add(case.id)
    return cases


async def run_case(
    client: FireworksClient, pin: ModelPin, case: Case, *, trials: int, max_tool_rounds: int
) -> CaseResult:
    if trials < 1:
        # Zero trials passes every case with nothing run.
        raise CaseError(f"trials must be at least 1, not {trials}")
    results = [
        await _trial(client, pin, case, max_tool_rounds=max_tool_rounds) for _ in range(trials)
    ]
    return CaseResult(case=case, trials=tuple(results))


def metered(settings: Settings) -> FireworksClient:
    return _MeteredClient(settings)


def grade(
    case: Case,
    *,
    calls: tuple[str, ...],
    staged: tuple[str, ...],
    reply: str,
    arguments: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Every expectation the trial missed. Empty means it passed.

    ``arguments`` maps a tool to the model's arguments on its first call
    that the broker staged or ran. A call it refused, such as an ``at``
    sent before ``get_time``, is not what the person would get.
    """
    expect = case.expect
    failures: list[str] = []
    lowered = reply.lower()
    failures += [f"did not call {name}" for name in expect.calls if name not in calls]
    failures += [f"called {name}" for name in expect.no_calls if name in calls]
    failures += [f"did not stage {name}" for name in expect.staged if name not in staged]
    failures += [f"staged {name}" for name in expect.no_staged if name in staged]
    failures += [
        f"reply lacks {text!r}" for text in expect.reply_contains if text.lower() not in lowered
    ]
    failures += [
        f"reply contains {text!r}" for text in expect.reply_excludes if text.lower() in lowered
    ]
    if expect.max_tool_calls is not None and len(calls) > expect.max_tool_calls:
        failures.append(f"{len(calls)} tool calls > {expect.max_tool_calls}")
    for tool, wanted in expect.arguments.items():
        sent = (arguments or {}).get(tool)
        if sent is None:
            failures.append(f"no {tool} call was staged or run to grade")
            continue
        failures += _argument_misses(tool, wanted, sent)
    return tuple(failures)


def accepted_arguments(messages: Sequence[Message]) -> dict[str, dict[str, Any]]:
    """The model's arguments on the first call of each tool the broker staged or ran.

    A refusal comes back as an ``error`` object, so a call whose result has
    one never reached the person and is passed over for the next.
    """
    results = {
        message.tool_call_id: message.content for message in messages if message.role == "tool"
    }
    first: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.role != "assistant":
            continue
        for call in message.tool_calls:
            if call.name in first or _refused(results.get(call.id)):
                continue
            try:
                parsed: object = json.loads(call.arguments) if call.arguments else {}
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                first[call.name] = parsed
    return first


async def _trial(
    client: FireworksClient, pin: ModelPin, case: Case, *, max_tool_rounds: int
) -> Trial:
    agent = AgentSpec(
        name=_AGENT,
        binding=Binding(channel="local", peer="eval"),
        reads=("own", "shared"),
        writes=("own", "shared"),
        tools=case.tools,
        web_access=case.web_access,
    )
    clock = case.clock()
    catalog = builtin_catalog()
    for name in catalog.names():
        catalog.bind(name, _fixture(case.fixtures.get(name, _DEFAULT_RESULT)))
    prompt = system_prompt_for(agent)
    if clock is not None:
        # The production handler and prompt, read on the case's stopped clock.
        catalog.bind("get_time", bind_clock(case.timezone, now=clock.now))
        prompt = system_prompt_for(agent, timezone=case.timezone, now=clock.now)
        if case.jobs:
            listing = job_list_result(
                sorted(case.jobs, key=lambda job: job.next_run_at), clock=clock
            )
            catalog.bind(JOB_LIST, _fixture(listing))
    # Staging fixes a named time on this clock and finds the job a
    # reschedule moves, as the channel's broker does.
    broker = ToolBroker(
        catalog,
        BrokerPolicy.defaults(),
        store=_CaseJobs(case.jobs) if case.jobs else None,
        clock=clock,
    )
    messages = [Message(role="system", content=prompt), *case.history]
    surface = broker.for_agent(agent, Scope.from_history(messages))
    messages.append(Message(role="user", content=case.user, created_at=time.time()))
    turn = len(messages)
    if isinstance(client, _MeteredClient):
        client.prompt_tokens = client.completion_tokens = 0
    started = time.monotonic()
    error: str | None = None
    try:
        await run_turn(client, pin, messages, surface, max_tool_rounds=max_tool_rounds)
    except AssistAIError as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = (time.monotonic() - started) * 1000
    calls = tuple(
        call.name
        for message in messages
        if message.role == "assistant"
        for call in message.tool_calls
    )
    staged = tuple(call.name for call in surface.take_staged())
    reply = last_assistant_text(messages)
    arguments = accepted_arguments(messages[turn:])
    failures = grade(case, calls=calls, staged=staged, reply=reply, arguments=arguments)
    if error is not None:
        failures = (error, *failures)
    metered_client = client if isinstance(client, _MeteredClient) else None
    return Trial(
        passed=not failures,
        failures=failures,
        calls=calls,
        staged=staged,
        reply=reply,
        prompt_tokens=metered_client.prompt_tokens if metered_client else 0,
        completion_tokens=metered_client.completion_tokens if metered_client else 0,
        latency_ms=elapsed,
        arguments=arguments,
    )


def _fixture(content: str) -> ToolHandler:
    async def handler(_arguments: dict[str, Any]) -> str:
        return content

    return handler


class _CaseJobs:
    """The case's stored jobs, found the way the store finds them."""

    def __init__(self, jobs: Sequence[Job]) -> None:
        self._jobs = tuple(jobs)

    def find_job(
        self, agent: str, *, name: str | None = None, job_id: str | None = None
    ) -> Job | None:
        if name is None and job_id is None:
            return None
        for job in self._jobs:
            if job.agent != agent or job.cancelled_at is not None:
                continue
            if job_id is not None and job.id != job_id:
                continue
            if name is not None and job.name.lower() != name.lower():
                continue
            return job
        return None


def _refused(content: str | None) -> bool:
    if content is None:
        return True
    try:
        parsed: object = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and "error" in parsed


def _argument_misses(tool: str, wanted: ArgumentExpect, sent: Mapping[str, Any]) -> list[str]:
    misses: list[str] = []
    for path, expected in wanted.equals.items():
        actual = _at_path(sent, path)
        if actual is _MISSING:
            misses.append(f"{tool} {path} is absent, expected {expected!r}")
        elif not _same(actual, expected):
            misses.append(f"{tool} {path} is {actual!r}, expected {expected!r}")
    misses += [
        f"{tool} has {path} = {_at_path(sent, path)!r}, expected none"
        for path in wanted.absent
        if _at_path(sent, path) is not _MISSING
    ]
    if wanted.any_of and not any(
        all(_same(_at_path(sent, path), value) for path, value in option.items())
        for option in wanted.any_of
    ):
        options = " or ".join(
            ", ".join(f"{path} = {value!r}" for path, value in option.items())
            for option in wanted.any_of
        )
        misses.append(f"{tool} matches none of: {options}")
    return misses


def _at_path(arguments: Mapping[str, Any], path: str) -> object:
    """The value at a dotted path. A JSON null counts as absent, as the parsers read it."""
    value: object = arguments
    for key in path.split("."):
        if not isinstance(value, Mapping) or value.get(key) is None:
            return _MISSING
        value = value[key]
    return value


def _same(actual: object, expected: Scalar) -> bool:
    """Equal as the parsers read it: a day word in any case, but true is not 1."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip().casefold() == expected.casefold()
    return isinstance(actual, int | float) and actual == expected


def _parse_case(path: Path, *, known: frozenset[str], default: AgentSpec) -> Case:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CaseError(f"{path.name}: {exc}") from exc
    case_id = raw.get("id")
    if case_id != path.stem:
        raise CaseError(f"{path.name}: id must equal the file name ({path.stem})")
    user = raw.get("user")
    if not isinstance(user, str) or not user.strip():
        raise CaseError(f"{path.name}: user must be a non-empty string")
    tools = tuple(raw.get("tools", default.tools))
    expect_raw = raw.get("expect", {})
    expect = Expect(
        calls=tuple(expect_raw.get("calls", ())),
        no_calls=tuple(expect_raw.get("no_calls", ())),
        staged=tuple(expect_raw.get("staged", ())),
        no_staged=tuple(expect_raw.get("no_staged", ())),
        reply_contains=tuple(expect_raw.get("reply_contains", ())),
        reply_excludes=tuple(expect_raw.get("reply_excludes", ())),
        max_tool_calls=expect_raw.get("max_tool_calls"),
        arguments=_parse_arguments(expect_raw.get("arguments", {}), where=path.name),
    )
    fixtures = {str(name): str(value) for name, value in raw.get("fixtures", {}).items()}
    timezone, now = _parse_clock(raw, where=path.name)
    jobs = _parse_jobs(raw.get("jobs", []), timezone=timezone, now=now, where=path.name)
    if now is not None and "get_time" in fixtures:
        raise CaseError(f"{path.name}: a case with now reads get_time from it, not a fixture")
    if jobs and JOB_LIST in fixtures:
        raise CaseError(f"{path.name}: job_list is written from jobs, not a fixture")
    named = {
        *tools,
        *expect.calls,
        *expect.no_calls,
        *expect.staged,
        *expect.no_staged,
        *expect.arguments,
        *fixtures,
    }
    unknown = sorted(name for name in named if name not in known)
    if unknown:
        raise CaseError(f"{path.name}: unknown tools {unknown}")
    history: list[Message] = []
    for item in raw.get("history", ()):
        role = item.get("role")
        if role not in _HISTORY_ROLES:
            raise CaseError(f"{path.name}: history role must be user or assistant")
        history.append(
            Message(
                role=role,
                content=str(item.get("content", "")),
                untrusted=bool(item.get("untrusted", False)),
                created_at=time.time(),
            )
        )
    pass_rate = float(raw.get("pass_rate", 1.0))
    if not 0 < pass_rate <= 1:
        raise CaseError(f"{path.name}: pass_rate must be in (0, 1]")
    return Case(
        id=case_id,
        user=user,
        tools=tools,
        web_access=bool(raw.get("web_access", default.web_access)),
        expect=expect,
        fixtures=fixtures,
        history=tuple(history),
        tags=tuple(raw.get("tags", ())),
        pass_rate=pass_rate,
        timezone=timezone,
        now=now,
        jobs=jobs,
    )


def _parse_clock(raw: dict[str, Any], *, where: str) -> tuple[str, float | None]:
    """The case's household zone and the instant its clock is stopped at."""
    timezone, now = raw.get("timezone"), raw.get("now")
    if timezone is None and now is None:
        return "", None
    if not isinstance(timezone, str) or now is None:
        raise CaseError(f"{where}: now and timezone go together")
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise CaseError(f"{where}: unknown timezone {timezone!r}") from exc
    return timezone, _local_instant(now, zone, field="now", where=where)


def _local_instant(raw: object, zone: ZoneInfo, *, field: str, where: str) -> float:
    # A local date-time is read on the household's wall, as a person names
    # it. An offset would let a case's clock disagree with its zone.
    if not isinstance(raw, dt.datetime) or raw.tzinfo is not None:
        raise CaseError(f"{where}: {field} must be a local date-time such as 2026-10-30T15:04:00")
    try:
        return wall_time(raw, zone).timestamp()
    except JobError as exc:
        raise CaseError(f"{where}: {field}: {exc}") from exc


def _parse_jobs(raw: object, *, timezone: str, now: float | None, where: str) -> tuple[Job, ...]:
    """Stored jobs the case starts with. ``job_list`` and a reschedule read these."""
    if not isinstance(raw, list):
        raise CaseError(f"{where}: jobs must be an array of tables")
    if not raw:
        return ()
    if now is None:
        raise CaseError(f"{where}: jobs need now and timezone")
    zone = ZoneInfo(timezone)
    jobs: list[Job] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise CaseError(f"{where}: jobs must be an array of tables")
        unknown = sorted(set(item) - {"name", "prompt", "next_run", "every_days", "verbatim"})
        if unknown:
            raise CaseError(
                f"{where}: jobs take name, prompt, next_run, every_days, verbatim, not {unknown}"
            )
        name, prompt = item.get("name"), item.get("prompt")
        if not isinstance(name, str) or not name or not isinstance(prompt, str) or not prompt:
            raise CaseError(f"{where}: each job needs a name and a prompt")
        every_days = item.get("every_days")
        if every_days is not None and (
            isinstance(every_days, bool) or not isinstance(every_days, int) or every_days < 1
        ):
            raise CaseError(f"{where}: every_days must be a whole number of days, at least 1")
        verbatim = item.get("verbatim", True)
        if not isinstance(verbatim, bool):
            raise CaseError(f"{where}: verbatim must be true or false")
        first = _local_instant(item.get("next_run"), zone, field="next_run", where=where)
        once = every_days is None
        every_seconds = (every_days or 1) * DAY_SECONDS
        # The schedule a whole-day row is given on the gateway's start, so the
        # case row reads as the migrated mgmplus row does.
        schedule = schedule_from_seconds(
            zone=zone, first_run=first, every_seconds=every_seconds, once=once
        )
        jobs.append(
            Job(
                id=f"j{index:08x}",
                agent=_AGENT,
                kind="schedule",
                name=name,
                prompt=prompt,
                every_seconds=every_seconds,
                ttl_seconds=None,
                created_at=now,
                expires_at=None,
                next_run_at=first,
                last_run_at=None,
                cancelled_at=None,
                once=once,
                verbatim=verbatim,
                zone=timezone,
                anchor_at=schedule.anchor_at,
                every_days=schedule.every_days,
                at_local=schedule.at_local,
            )
        )
    return tuple(jobs)


def _parse_arguments(raw: object, *, where: str) -> dict[str, ArgumentExpect]:
    """``[expect.arguments.<tool>]`` tables, each with equals, absent, or any_of."""
    if not isinstance(raw, dict):
        raise CaseError(f"{where}: expect.arguments must be a table of tools")
    parsed: dict[str, ArgumentExpect] = {}
    for tool, table in raw.items():
        label = f"{where}: expect.arguments.{tool}"
        if not isinstance(table, dict):
            raise CaseError(f"{label} must be a table")
        unknown = sorted(set(table) - _ARGUMENT_KEYS)
        if unknown:
            raise CaseError(f"{label} takes equals, absent, and any_of, not {unknown}")
        equals = _flat_values(table.get("equals", {}), label=f"{label}.equals")
        absent_raw = table.get("absent", [])
        if not isinstance(absent_raw, list) or not all(isinstance(p, str) for p in absent_raw):
            raise CaseError(f"{label}.absent must be a list of dotted paths")
        absent = tuple(_checked_path(p, label=f"{label}.absent") for p in absent_raw)
        options_raw = table.get("any_of", [])
        if not isinstance(options_raw, list):
            raise CaseError(f"{label}.any_of must be an array of tables")
        any_of = tuple(_flat_values(option, label=f"{label}.any_of") for option in options_raw)
        if any(not option for option in any_of):
            raise CaseError(f"{label}.any_of has an empty table, which would always match")
        if not equals and not absent and not any_of:
            raise CaseError(f"{label} expects nothing")
        both = sorted(set(equals) & set(absent))
        if both:
            raise CaseError(f"{label} wants {both} both set and absent")
        parsed[str(tool)] = ArgumentExpect(equals=equals, absent=absent, any_of=any_of)
    return parsed


def _flat_values(raw: object, *, label: str, prefix: str = "") -> dict[str, Scalar]:
    """Path to value. ``at.hour = 8`` and ``"at.hour" = 8`` are the same expectation."""
    if not isinstance(raw, dict):
        raise CaseError(f"{label} must be a table of dotted paths")
    flat: dict[str, Scalar] = {}
    for key, value in raw.items():
        path = _checked_path(f"{prefix}{key}", label=label)
        if isinstance(value, dict):
            found = _flat_values(value, label=label, prefix=f"{path}.")
        elif isinstance(value, str | int | float | bool):
            found = {path: value}
        else:
            raise CaseError(f"{label}: {path} must be a string, number, or boolean")
        repeated = sorted(set(found) & set(flat))
        if repeated:
            raise CaseError(f"{label}: {repeated} given twice")
        flat.update(found)
    return flat


def _checked_path(path: str, *, label: str) -> str:
    if not _PATH.fullmatch(path):
        raise CaseError(f"{label}: {path!r} is not a dotted path such as at.hour")
    return path
