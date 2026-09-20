"""Route inbound Signal DMs through policy and the Fireworks loop."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Literal

import structlog

from assistai.agents import AgentSpec, Household, system_prompt_for
from assistai.broker import BoundSurface, Scope, ToolBroker
from assistai.config import Settings
from assistai.conversation import apply_window, last_assistant_text, trusted_context
from assistai.errors import AssistAIError, StoreError
from assistai.inference.client import FireworksClient
from assistai.inference.loop import run_turn
from assistai.inference.types import Message, ToolCall
from assistai.jobs import Job, JobOutcome, bind_jobs, is_nothing, job_user_prompt
from assistai.manifest import Manifest
from assistai.relay import RELAY_TOOL, RelayError, bind_relay
from assistai.scheduler import NotifyStatus
from assistai.signal.client import SignalTransport
from assistai.signal.envelopes import InboundText, parse_inbound
from assistai.signal.numbers import is_uuid
from assistai.signal.policy import AccessPolicy
from assistai.signal.ratelimit import RateLimiter
from assistai.staging import Proposal, format_executed, format_proposal, parse_decision
from assistai.store import Store

log = structlog.get_logger(__name__)

_UNAVAILABLE = "I could not reach the model just now. Try again in a moment."
_SAVE_FAILED = "I could not save that just now. Try again in a moment."
_NOT_CONFIGURED = "Inference is not configured on this gateway yet."
_PAIRING_HINT = (
    "This number is not authorized.\n\n"
    "Give this pairing code to the operator: {code}\n"
    "They reply /approve {code} from an operator phone."
)
_APPROVE_OK = (
    "Approved {number}. They can message the bot, but they only reach an agent "
    "once one is bound to their number in config/assistai.toml."
)
_APPROVE_BAD = "No pending pairing matches that code (expired or already used)."
_APPROVE_DENIED = "Only an operator can approve pairing codes."
_TOO_LONG = "That message is too long ({actual} characters, limit {limit}). Send a shorter one."
_TOO_FAST = "You are sending faster than I can answer. Try again in a minute."
_UNBOUND = (
    "This number is allowed to message the bot, but no agent is bound to it. "
    "The operator has to add a binding in config/assistai.toml."
)
_DISCARDED = "Discarded."
_EXPIRED = "That proposal expired."


class SignalChannel:
    """Inbound DMs and job reports over Signal.

    Turns are serialized per sender, so a slow model call for one person does
    not block the other. Job sends use a separate meter.
    """

    def __init__(
        self,
        settings: Settings,
        signal: SignalTransport,
        *,
        policy: AccessPolicy,
        fireworks: FireworksClient | None,
        manifest: Manifest | None,
        household: Household,
        broker: ToolBroker,
        store: Store,
    ) -> None:
        self._settings = settings
        self._signal = signal
        self._policy = policy
        self._fireworks = fireworks
        self._manifest = manifest
        self._household = household
        self._broker = broker
        self._store = store
        broker.bind(
            RELAY_TOOL,
            bind_relay(
                household=household,
                signal=signal,
                inject=self._inject_relay,
            ),
        )
        for name, handler in bind_jobs(store).items():
            broker.bind(name, handler)
        self._histories: dict[str, list[Message]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._history_locks: dict[str, asyncio.Lock] = {}
        self._uuid_numbers: dict[str, str] = {}
        self._turns = RateLimiter(settings.signal_rate_limit_per_minute, 60.0)
        # Strangers get a much smaller budget: each reply is outbound traffic
        # from the bot number, which Signal counts against us, not them.
        self._pairing = RateLimiter(settings.signal_pairing_replies_per_hour, 3600.0)
        self._job_sends = RateLimiter(settings.signal_job_messages_per_hour, 3600.0)
        self._slots = asyncio.Semaphore(settings.max_concurrent_turns)
        broker.use_store(store)

    async def run(self, stop: asyncio.Event) -> None:
        """Consume receive frames until shutdown.

        Each message is dispatched as its own task so a slow model call for one
        person does not stall everyone else; the per-sender lock still keeps a
        single conversation strictly ordered.
        """
        tasks: set[asyncio.Task[None]] = set()
        try:
            async for payload in self._signal.receive(stop):
                inbound = parse_inbound(payload)
                if inbound is None:
                    log.debug("signal.ignored_frame", payload=_frame_summary(payload))
                    continue
                log.info(
                    "signal.inbound",
                    sender=inbound.sender,
                    chars=len(inbound.text),
                    attachments=len(inbound.attachments),
                )
                task = asyncio.create_task(self._guarded(inbound))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        finally:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _guarded(self, inbound: InboundText) -> None:
        try:
            await self.handle(inbound)
        except Exception:
            log.exception("signal.handle_failed", sender=inbound.sender)

    async def handle(self, inbound: InboundText) -> None:
        """Policy, then either pairing, approve, or a model turn."""
        resolved = await self._resolve_sender(inbound)
        if resolved is None:
            return
        inbound = resolved
        if self._policy.decide(inbound.sender) == "unknown":
            await self._handle_unknown(inbound)
            return
        async with self._lock_for(inbound.sender):
            await self._handle_allowed(inbound)

    async def _handle_allowed(self, inbound: InboundText) -> None:
        code = self._policy.parse_approve(inbound.text)
        if code is not None:
            await self._handle_approve(inbound, code)
            return
        # Resolve the agent before spending any budget, so a number with no
        # binding is told that rather than being rate limited into confusion.
        agent = self._household.agent_for_signal_dm(inbound.sender)
        if agent is None:
            log.info("signal.unbound", sender=inbound.sender)
            # Still metered: the answer is free of model cost but not of Signal
            # traffic, and a loop on the other end should not be amplified.
            if self._turns.allow(inbound.sender):
                await self._safe_send(inbound.sender, _UNBOUND)
            return
        # A message that is only an attachment still needs a turn so the model
        # can propose a relay. The binary is not downloaded this phase.
        if inbound.attachments and not inbound.text:
            names = ", ".join(item.name for item in inbound.attachments)
            inbound = InboundText(
                sender=inbound.sender,
                text=(
                    f"[attachment: {names} — paste a link to relay it; file send is not built yet]"
                ),
                timestamp=inbound.timestamp,
                attachments=inbound.attachments,
            )
        limit = self._settings.signal_max_inbound_chars
        if len(inbound.text) > limit:
            log.info("signal.rejected_long", sender=inbound.sender, chars=len(inbound.text))
            await self._safe_send(
                inbound.sender, _TOO_LONG.format(actual=len(inbound.text), limit=limit)
            )
            return
        if not self._turns.allow(inbound.sender):
            log.warning("signal.rate_limited", sender=inbound.sender)
            await self._safe_send(inbound.sender, _TOO_FAST)
            return
        async with self._slots:
            if await self._handle_decision(inbound, agent):
                return
            await self._converse(inbound, agent)

    async def run_job(self, agent: AgentSpec, job: Job) -> JobOutcome:
        """Report-only turn for a due job. Does not send; the runner decides."""
        if self._fireworks is None or self._manifest is None:
            raise AssistAIError(_NOT_CONFIGURED)
        # Same order as an inbound turn: peer, then slot, then history. A relay
        # confirm holds a slot while it waits for the other person's history
        # lock, so taking history before the slot can deadlock both of them.
        async with self._lock_for(agent.binding.peer):
            async with self._slots:
                async with self._history_lock(agent.name):
                    try:
                        messages = self._history(agent.name, system_prompt_for(agent))
                    except StoreError:
                        log.exception("store.history_unreadable", agent=agent.name)
                        raise
                    surface: BoundSurface = self._broker.for_agent(
                        agent, Scope.from_history(messages), report_only=True
                    )
                    scratch = [
                        *trusted_context(messages),
                        Message(
                            role="user",
                            content=job_user_prompt(job),
                            created_at=time.time(),
                        ),
                    ]
                    await run_turn(
                        self._fireworks,
                        self._manifest.primary,
                        scratch,
                        surface,
                        max_tool_rounds=self._settings.max_tool_rounds,
                    )
                    text = last_assistant_text(scratch) or ""
                    # Label this report for what this run fetched. Inherited
                    # taint stays on the messages that carry it, so it can
                    # still age out of the window on schedule.
                    return JobOutcome(
                        text=text,
                        found=not is_nothing(text),
                        untrusted=surface.fetched_untrusted,
                    )

    def can_notify_job(self, agent: AgentSpec) -> bool:
        return self._job_sends.would_allow(agent.name)

    async def notify_job(self, agent: AgentSpec, text: str, *, untrusted: bool) -> NotifyStatus:
        """Send a job report. History is persisted separately after delivery is recorded."""
        _ = untrusted
        if not self._job_sends.would_allow(agent.name):
            return "rate_limited"
        try:
            await self._signal.send(agent.binding.peer, text)
        except AssistAIError:
            log.exception("signal.send_failed", recipient=agent.binding.peer)
            return "failed"
        self._job_sends.record(agent.name)
        return "sent"

    async def persist_job_report(self, agent: AgentSpec, text: str, *, untrusted: bool) -> bool:
        """Append a job report that already went out over Signal."""
        async with self._history_lock(agent.name):
            self._histories.pop(agent.name, None)
            try:
                messages = self._history(agent.name, system_prompt_for(agent))
            except StoreError:
                log.exception("store.history_unreadable", agent=agent.name)
                return False
            messages.append(
                Message(
                    role="assistant",
                    content=text,
                    untrusted=untrusted,
                    created_at=time.time(),
                )
            )
            saved = self._window_and_persist(agent.name, messages)
            if not saved:
                # The append mutated the cached list. Leave it there and a later
                # turn persists a report the store never accepted; the retry
                # then appends it a second time.
                self._histories.pop(agent.name, None)
            return saved

    async def _converse(self, inbound: InboundText, agent: AgentSpec) -> None:
        if self._fireworks is None or self._manifest is None:
            await self._safe_send(inbound.sender, _NOT_CONFIGURED)
            return
        async with self._history_lock(agent.name):
            try:
                messages = self._history(agent.name, system_prompt_for(agent))
            except StoreError:
                log.exception("store.history_unreadable", agent=agent.name)
                await self._safe_send(inbound.sender, _UNAVAILABLE)
                return
            # Taint carries across turns: untrusted text fetched last message is
            # still in this history, so a privileged sink must stay refused.
            surface: BoundSurface = self._broker.for_agent(agent, Scope.from_history(messages))
            # A failed turn can leave an assistant tool-call with no result, which
            # the provider rejects forever after. Roll the whole turn back instead.
            baseline = len(messages)
            messages.append(Message(role="user", content=inbound.text, created_at=time.time()))
            try:
                await run_turn(
                    self._fireworks,
                    self._manifest.primary,
                    messages,
                    surface,
                    max_tool_rounds=self._settings.max_tool_rounds,
                )
            except AssistAIError:
                log.exception("signal.turn_failed", sender=inbound.sender, agent=agent.name)
                del messages[baseline:]
                await self._safe_send(inbound.sender, _UNAVAILABLE)
                return
            reply = last_assistant_text(messages)
            if not reply:
                reply = _UNAVAILABLE
            staged = surface.take_staged()
            proposal: Proposal | Literal["keep"] = "keep"
            if staged:
                now = time.time()
                preview = format_proposal(staged, tainted=surface.tainted)
                messages.append(Message(role="assistant", content=preview, created_at=now))
                reply = preview
                proposal = Proposal(
                    agent=agent.name,
                    calls=staged,
                    tainted=surface.tainted,
                    created_at=now,
                    expires_at=now + self._settings.staging_ttl_seconds,
                )
            if not self._window_and_persist(agent.name, messages, proposal=proposal):
                del messages[baseline:]
                if staged:
                    self._abandon_live_proposal(agent.name)
                await self._safe_send(inbound.sender, _SAVE_FAILED)
                return
        await self._safe_send(inbound.sender, reply)

    async def _handle_decision(self, inbound: InboundText, agent: AgentSpec) -> bool:
        """Confirm or discard a live proposal. False means this is a normal turn."""
        decision = parse_decision(inbound.text)
        if decision is None:
            return False
        try:
            pending, expired = self._store.take_proposal(agent.name)
        except StoreError:
            log.exception("store.proposal_unreadable", agent=agent.name)
            await self._safe_send(inbound.sender, _UNAVAILABLE)
            return True
        if expired:
            reply = _EXPIRED if decision == "confirm" else _DISCARDED
            await self._safe_send(inbound.sender, reply)
            return True
        if pending is None:
            return False
        if decision == "confirm":
            await self._commit_proposal(inbound, agent, pending)
        else:
            await self._reject_proposal(inbound, agent)
        return True

    async def _commit_proposal(
        self, inbound: InboundText, agent: AgentSpec, pending: Proposal
    ) -> None:
        # Load history before consuming the proposal. A yes that cannot
        # proceed must still be retryable. Release the history lock before
        # handlers run so a relay inject can take the recipient's lock
        # without deadlocking two confirms against each other.
        async with self._history_lock(agent.name):
            try:
                messages = self._history(agent.name, system_prompt_for(agent))
            except StoreError:
                log.exception("store.history_unreadable", agent=agent.name)
                await self._safe_send(inbound.sender, _UNAVAILABLE)
                return
            # Drop the proposal before the handlers run. A later yes must not fire
            # the same stored calls twice if persist fails after the side effects.
            try:
                self._store.clear_proposal(agent.name)
            except StoreError:
                log.exception("store.proposal_clear_failed", agent=agent.name)
                await self._safe_send(inbound.sender, _SAVE_FAILED)
                return
            scope = Scope.from_history(messages)
        surface: BoundSurface = self._broker.for_agent(agent, scope)
        results: list[tuple[ToolCall, str]] = []
        for call in pending.calls:
            result = await surface.commit(call)
            results.append((call, result.content))
        now = time.time()
        summary = format_executed(results)
        async with self._history_lock(agent.name):
            # Reload: a relay injected into this conversation while handlers
            # ran must not be overwritten by a stale in-memory copy.
            self._histories.pop(agent.name, None)
            try:
                messages = self._history(agent.name, system_prompt_for(agent))
            except StoreError:
                log.error("store.save_failed_after_commit", agent=agent.name)
                await self._safe_send(inbound.sender, summary)
                return
            messages.append(Message(role="user", content=inbound.text, created_at=now))
            messages.append(Message(role="assistant", content=summary, created_at=now))
            if not self._window_and_persist(agent.name, messages):
                log.error("store.save_failed_after_commit", agent=agent.name)
        log.info("staging.committed", agent=agent.name, calls=len(pending.calls))
        await self._safe_send(inbound.sender, summary)

    async def _reject_proposal(self, inbound: InboundText, agent: AgentSpec) -> None:
        async with self._history_lock(agent.name):
            try:
                messages = self._history(agent.name, system_prompt_for(agent))
            except StoreError:
                log.exception("store.history_unreadable", agent=agent.name)
                await self._safe_send(inbound.sender, _UNAVAILABLE)
                return
            now = time.time()
            messages.append(Message(role="user", content=inbound.text, created_at=now))
            messages.append(Message(role="assistant", content=_DISCARDED, created_at=now))
            if not self._window_and_persist(agent.name, messages, proposal=None):
                await self._safe_send(inbound.sender, _SAVE_FAILED)
                return
        log.info("staging.discarded", agent=agent.name)
        await self._safe_send(inbound.sender, _DISCARDED)

    def _abandon_live_proposal(self, agent_name: str) -> None:
        """A failed staging save must not leave an older proposal confirmable."""
        try:
            self._store.clear_proposal(agent_name)
        except StoreError:
            log.exception("store.proposal_clear_failed", agent=agent_name)

    def _window_and_persist(
        self,
        agent_name: str,
        messages: list[Message],
        *,
        proposal: Proposal | Literal["keep"] | None = "keep",
    ) -> bool:
        windowed = messages[:]
        apply_window(
            windowed,
            keep=self._settings.history_keep,
            max_age_seconds=self._settings.history_max_age_seconds,
        )
        try:
            self._store.persist(agent_name, windowed, proposal=proposal)
        except StoreError:
            log.exception("store.save_failed", agent=agent_name)
            return False
        messages[:] = windowed
        return True

    def _history(self, agent_name: str, prompt: str) -> list[Message]:
        if agent_name not in self._histories:
            loaded = self._store.load_history(agent_name)
            messages = [Message(role="system", content=prompt), *loaded]
            windowed = messages[:]
            apply_window(
                windowed,
                keep=self._settings.history_keep,
                max_age_seconds=self._settings.history_max_age_seconds,
            )
            if len(windowed) < len(messages):
                try:
                    self._store.save_history(agent_name, windowed)
                except StoreError:
                    log.exception("store.save_failed", agent=agent_name)
                    self._histories[agent_name] = messages
                    return messages
                messages = windowed
            self._histories[agent_name] = messages
        return self._histories[agent_name]

    async def _handle_approve(self, inbound: InboundText, code: str) -> None:
        try:
            approved = self._policy.approve(code, approver=inbound.sender)
        except PermissionError:
            log.warning("signal.approve_denied", sender=inbound.sender)
            await self._safe_send(inbound.sender, _APPROVE_DENIED)
            return
        except StoreError:
            log.exception("store.admit_failed", sender=inbound.sender)
            await self._safe_send(inbound.sender, _SAVE_FAILED)
            return
        reply = _APPROVE_OK.format(number=approved) if approved else _APPROVE_BAD
        await self._safe_send(inbound.sender, reply)

    async def _handle_unknown(self, inbound: InboundText) -> None:
        if self._settings.signal_dm_policy == "allowlist":
            log.info("signal.denied", sender=inbound.sender, reason="not_allowlisted")
            return
        if not self._pairing.allow(inbound.sender):
            log.warning("signal.pairing_throttled", sender=inbound.sender)
            return
        code = self._policy.request_pair(inbound.sender)
        await self._safe_send(inbound.sender, _PAIRING_HINT.format(code=code))

    async def _resolve_sender(self, inbound: InboundText) -> InboundText | None:
        """Map a privacy UUID onto an E.164 number the allowlist understands."""
        if inbound.sender.startswith("+"):
            return inbound
        if not is_uuid(inbound.sender):
            log.warning("signal.unresolved_sender", source=inbound.sender)
            return None
        cached = self._uuid_numbers.get(inbound.sender)
        if cached is not None:
            return replace(inbound, sender=cached)
        found = await self._signal.number_for_uuid(inbound.sender)
        if found is None:
            mapped = set(self._uuid_numbers.values())
            allow = [
                number
                for number in self._settings.allow_from
                if number != self._settings.signal_account and number not in mapped
            ]
            if len(allow) == 1:
                found = allow[0]
        if found is None:
            log.warning("signal.unresolved_sender", source=inbound.sender)
            return None
        self._uuid_numbers[inbound.sender] = found
        log.info("signal.resolved_uuid", uuid=inbound.sender, number=found)
        return replace(inbound, sender=found)

    async def _inject_relay(self, recipient: AgentSpec, record: str) -> None:
        """Append an untrusted relay record under the recipient's history lock."""
        async with self._history_lock(recipient.name):
            self._append_untrusted(recipient.name, record)

    def _append_untrusted(self, agent_name: str, content: str) -> None:
        """Caller holds ``_history_lock(agent_name)``. Window includes the system prompt."""
        agent = self._agent_named(agent_name)
        loaded = self._store.load_history(agent_name)
        messages = [
            Message(role="system", content=system_prompt_for(agent)),
            *loaded,
            Message(role="user", content=content, untrusted=True, created_at=time.time()),
        ]
        if not self._window_and_persist(agent_name, messages):
            raise StoreError(f"history for {agent_name} could not be saved")
        self._histories[agent_name] = messages

    def _agent_named(self, name: str) -> AgentSpec:
        for agent in self._household.agents:
            if agent.name == name:
                return agent
        raise RelayError(f"no agent named {name}")

    def _history_lock(self, agent_name: str) -> asyncio.Lock:
        lock = self._history_locks.get(agent_name)
        if lock is None:
            lock = asyncio.Lock()
            self._history_locks[agent_name] = lock
        return lock

    def _lock_for(self, sender: str) -> asyncio.Lock:
        lock = self._locks.get(sender)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[sender] = lock
        return lock

    async def _safe_send(self, recipient: str, text: str) -> None:
        try:
            await self._signal.send(recipient, text)
        except AssistAIError:
            log.exception("signal.send_failed", recipient=recipient)


def _frame_summary(payload: object) -> dict[str, object]:
    """Enough to see why a receive frame was ignored, without dumping the body."""
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    envelope = payload.get("envelope") if isinstance(payload.get("envelope"), dict) else payload
    if not isinstance(envelope, dict):
        return {"keys": sorted(payload.keys())[:12]}
    return {
        "keys": sorted(payload.keys())[:12],
        "source": (
            envelope.get("sourceNumber") or envelope.get("sourceUuid") or envelope.get("source")
        ),
        "has_data": "dataMessage" in envelope,
        "has_receipt": "receiptMessage" in envelope,
        "has_typing": "typingMessage" in envelope,
    }
