"""The security boundary. Agents never execute tools; they ask the broker.

A bound surface advertises only the agent's allowlist, refuses anything else
before the handler runs, and taints the conversation when an untrusted tool
returns. Once tainted, unstaged sinks are refused even if they sit on the
allowlist. Staging tools still propose: the handler does not run until the
person confirms the stored call. Taint outlives the turn because the untrusted
text does.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import structlog

from assistai.agents import AgentSpec, BrokerPolicy
from assistai.errors import JobError
from assistai.inference.tools import (
    GET_TIME_SPEC,
    ToolHandler,
    ToolRegistry,
    ToolResult,
    get_time,
)
from assistai.inference.types import Message, ToolCall, ToolSpec
from assistai.jobs import (
    JOB_CANCEL,
    JOB_CANCEL_SPEC,
    JOB_CREATE,
    JOB_CREATE_SPEC,
    JOB_LIST_SPEC,
    JOB_RESCHEDULE,
    JOB_RESCHEDULE_SPEC,
    Job,
    parse_cancel,
    parse_create,
    parse_reschedule,
)
from assistai.relay import RELAY_SPEC, RELAY_TOOL, RelayError, parse_body, using_agent

log = structlog.get_logger(__name__)

DenyReason = Literal["unknown", "acl", "web_access", "taint"]
AuditAction = Literal["allow", "deny", "stage"]

# The Pi runs this for weeks. Keep enough audit history to explain a refusal
# without letting a long uptime grow the process.
AUDIT_LIMIT = 1000


class _JobLookup(Protocol):
    def find_job(
        self, agent: str, *, name: str | None = None, job_id: str | None = None
    ) -> Job | None: ...


@dataclass(frozen=True)
class ToolMeta:
    """Broker-only attributes. Never sent to the model."""

    trusted: bool = True
    sink: str | None = None
    web: bool = False
    staging: bool = False


@dataclass(frozen=True)
class AuditEvent:
    """One allow, deny, or stage. Tests read this; production logs it."""

    agent: str
    tool: str
    action: AuditAction
    reason: DenyReason | None = None


@dataclass
class Scope:
    """Taint for one conversation, not one turn.

    Untrusted text stays in history after the turn that fetched it, so a scope
    that reset each turn would let the *next* message reach a privileged sink
    with the attacker's instructions still in context.
    """

    tainted: bool = False

    @classmethod
    def from_history(cls, messages: Iterable[Message]) -> Scope:
        return cls(tainted=any(message.untrusted for message in messages))


class ToolCatalog:
    """Every tool the process knows how to run, plus the metadata the broker needs."""

    def __init__(self) -> None:
        self._registry = ToolRegistry()
        self._meta: dict[str, ToolMeta] = {}

    def add(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        trusted: bool = True,
        sink: str | None = None,
        web: bool = False,
        staging: bool = False,
    ) -> None:
        self._registry.register(spec, handler)
        self._meta[spec.name] = ToolMeta(trusted=trusted, sink=sink, web=web, staging=staging)

    def bind(self, name: str, handler: ToolHandler) -> None:
        """Replace the handler for a tool already on the catalog."""
        spec = self._registry.spec(name)
        if spec is None:
            raise KeyError(name)
        self._registry.register(spec, handler)

    def names(self) -> frozenset[str]:
        return self._registry.names()

    def spec(self, name: str) -> ToolSpec | None:
        return self._registry.spec(name)

    def meta(self, name: str) -> ToolMeta:
        return self._meta.get(name, ToolMeta())

    async def execute(self, call: ToolCall) -> ToolResult:
        return await self._registry.execute(call)


class ToolBroker:
    """Builds a per-agent surface the loop can call."""

    def __init__(
        self, catalog: ToolCatalog, policy: BrokerPolicy, *, store: _JobLookup | None = None
    ) -> None:
        self._catalog = catalog
        self._policy = policy
        self._store = store
        self.audit: deque[AuditEvent] = deque(maxlen=AUDIT_LIMIT)

    def use_store(self, store: _JobLookup) -> None:
        self._store = store

    def for_agent(
        self, agent: AgentSpec, scope: Scope | None = None, *, report_only: bool = False
    ) -> BoundSurface:
        return BoundSurface(
            catalog=self._catalog,
            policy=self._policy,
            agent=agent,
            scope=scope or Scope(),
            record=self.record,
            report_only=report_only,
            store=self._store,
        )

    def bind(self, name: str, handler: ToolHandler) -> None:
        """Replace the live handler for a catalog tool."""
        self._catalog.bind(name, handler)

    def record(self, event: AuditEvent) -> None:
        self.audit.append(event)
        if event.action == "deny":
            log.warning(
                "broker.denied",
                agent=event.agent,
                tool=event.tool,
                reason=event.reason,
            )
            return
        if event.action == "stage":
            log.info("broker.staged", agent=event.agent, tool=event.tool)
            return
        log.info("broker.executed", agent=event.agent, tool=event.tool)


class BoundSurface:
    """The loop's view of one agent, scoped to one conversation."""

    def __init__(
        self,
        *,
        catalog: ToolCatalog,
        policy: BrokerPolicy,
        agent: AgentSpec,
        scope: Scope,
        record: Callable[[AuditEvent], None],
        report_only: bool = False,
        store: _JobLookup | None = None,
    ) -> None:
        self._catalog = catalog
        self._policy = policy
        self._agent = agent
        self._scope = scope
        self._record = record
        self._report_only = report_only
        self._store = store
        self._staged: list[ToolCall] = []
        self._fetched_untrusted = False

    @property
    def tainted(self) -> bool:
        return self._scope.tainted

    @property
    def fetched_untrusted(self) -> bool:
        """Untrusted output this surface fetched, ignoring taint it inherited.

        A job report labelled from inherited taint would re-taint the window on
        every run, so taint could never age out of a household that schedules
        anything.
        """
        return self._fetched_untrusted

    def take_staged(self) -> tuple[ToolCall, ...]:
        """Calls recorded this turn. Empty if the model did not stage."""
        staged = tuple(self._staged)
        self._staged.clear()
        return staged

    def specs(self) -> list[ToolSpec]:
        allowed: list[ToolSpec] = []
        for name in self._agent.tools:
            spec = self._catalog.spec(name)
            if spec is None:
                continue
            meta = self._catalog.meta(name)
            if meta.web and not self._agent.web_access:
                continue
            if self._report_only and _not_report_tool(name, meta):
                continue
            allowed.append(spec)
        return allowed

    async def execute(self, call: ToolCall) -> ToolResult:
        denied = self._deny_reason(call.name)
        if denied is not None:
            self._record(
                AuditEvent(agent=self._agent.name, tool=call.name, action="deny", reason=denied)
            )
            body = json.dumps({"error": "tool_denied", "name": call.name, "reason": denied})
            return ToolResult(body)
        meta = self._catalog.meta(call.name)
        if meta.staging:
            reason = _staging_error(call, self._agent, self._store, self._staged)
            if reason is not None:
                # Say why. The model has to tell the person whether to pick a
                # different name or a different job, and the commit-path
                # message never reaches them: staging refuses first.
                body = json.dumps(
                    {"error": "invalid_arguments", "name": call.name, "message": reason},
                    ensure_ascii=False,
                )
                return ToolResult(body)
            self._staged.append(call)
            self._record(AuditEvent(agent=self._agent.name, tool=call.name, action="stage"))
            body = json.dumps({"status": "staged", "name": call.name, "arguments": call.arguments})
            return ToolResult(body)
        return await self._run(call, meta)

    async def commit(self, call: ToolCall) -> ToolResult:
        """Run a previously staged call. The handler runs; it is not re-staged."""
        denied = self._deny_reason(call.name)
        if denied is not None:
            self._record(
                AuditEvent(agent=self._agent.name, tool=call.name, action="deny", reason=denied)
            )
            body = json.dumps({"error": "tool_denied", "name": call.name, "reason": denied})
            return ToolResult(body)
        return await self._run(call, self._catalog.meta(call.name))

    async def _run(self, call: ToolCall, meta: ToolMeta) -> ToolResult:
        with using_agent(self._agent):
            result = await self._catalog.execute(call)
        if not meta.trusted:
            self._scope.tainted = True
            self._fetched_untrusted = True
        self._record(AuditEvent(agent=self._agent.name, tool=call.name, action="allow"))
        return ToolResult(result.content, untrusted=not meta.trusted)

    def _deny_reason(self, name: str) -> DenyReason | None:
        if name not in self._catalog.names():
            return "unknown"
        if name not in self._agent.tools:
            return "acl"
        meta = self._catalog.meta(name)
        if self._report_only and _not_report_tool(name, meta):
            return "acl"
        if meta.web and not self._agent.web_access:
            return "web_access"
        if (
            self._scope.tainted
            and meta.sink is not None
            and meta.sink in self._policy.tainted_sinks_denied
            and not meta.staging
        ):
            return "taint"
        return None


def _not_report_tool(name: str, meta: ToolMeta) -> bool:
    """Jobs never file, draft, write, relay, or spawn more jobs."""
    if meta.staging or meta.sink is not None:
        return True
    return name.startswith("job_")


def _staging_error(
    call: ToolCall,
    agent: AgentSpec,
    store: _JobLookup | None = None,
    staged: Sequence[ToolCall] = (),
) -> str | None:
    """Why a staged call could not be stored, or None when it is executable as-is."""
    try:
        parsed: object = json.loads(call.arguments) if call.arguments else {}
        if not isinstance(parsed, dict):
            return "arguments must be a JSON object"
        if call.name == RELAY_TOOL:
            parse_body(parsed, from_name=agent.name)
            return None
        if call.name == JOB_CREATE:
            created = parse_create(parsed)
            existing = store.find_job(agent.name, name=created.name) if store is not None else None
            if existing is not None and not _staged_cancel_covers(existing, staged):
                return f"a job named {created.name!r} already exists"
            for other in staged:
                if other.name != JOB_CREATE:
                    continue
                previous: object = json.loads(other.arguments) if other.arguments else {}
                if not isinstance(previous, dict):
                    continue
                if parse_create(previous).name.lower() == created.name.lower():
                    return f"a job named {created.name!r} already exists"
            return None
        if call.name == JOB_RESCHEDULE:
            change = parse_reschedule(parsed)
            if store is None:
                return None
            found = store.find_job(agent.name, name=change.name, job_id=change.id)
            if found is None or _staged_cancel_covers(found, staged):
                return "no matching job"
            if found.ttl_seconds is not None and change.every_seconds > found.ttl_seconds:
                return "every_seconds is longer than the watch TTL"
            return None
        if call.name == JOB_CANCEL:
            name, job_id = parse_cancel(parsed)
            if store is None:
                return None
            found = store.find_job(agent.name, name=name, job_id=job_id)
            if found is None or _staged_cancel_covers(found, staged):
                return "no matching job"
            return None
    except (RelayError, JobError) as exc:
        return str(exc)
    except json.JSONDecodeError:
        return "arguments are not valid JSON"
    return None


def _staged_cancel_covers(job: Job, staged: Sequence[ToolCall]) -> bool:
    """True when an earlier call in this proposal already cancels ``job``."""
    for other in staged:
        if other.name != JOB_CANCEL:
            continue
        try:
            parsed: object = json.loads(other.arguments) if other.arguments else {}
            if not isinstance(parsed, dict):
                continue
            name, job_id = parse_cancel(parsed)
        except (JobError, json.JSONDecodeError):
            continue
        if job_id is not None and job_id == job.id:
            return True
        if name is not None and name.lower() == job.name.lower():
            return True
    return False


def builtin_catalog() -> ToolCatalog:
    """The tools this build actually implements.

    ``relay`` and the job tools are on the catalog so household ACLs can name
    them. Live handlers are bound when the Signal channel has a store.
    """
    catalog = ToolCatalog()
    catalog.add(GET_TIME_SPEC, get_time, trusted=True)

    async def _unbound(_arguments: dict[str, object]) -> str:
        raise RuntimeError("handler is not bound")

    catalog.add(
        RELAY_SPEC,
        _unbound,
        trusted=True,
        sink="message:other_peer",
        staging=True,
    )
    catalog.add(JOB_CREATE_SPEC, _unbound, trusted=True, staging=True)
    catalog.add(JOB_RESCHEDULE_SPEC, _unbound, trusted=True, staging=True)
    catalog.add(JOB_LIST_SPEC, _unbound, trusted=True)
    catalog.add(JOB_CANCEL_SPEC, _unbound, trusted=True, staging=True)
    return catalog
