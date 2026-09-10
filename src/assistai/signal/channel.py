"""Route inbound Signal DMs through policy and the Fireworks loop."""

from __future__ import annotations

import asyncio
import time

import structlog

from assistai.agents import AgentSpec, Household, system_prompt_for
from assistai.broker import BoundSurface, Scope, ToolBroker
from assistai.config import Settings
from assistai.conversation import apply_window, last_assistant_text
from assistai.errors import AssistAIError, StoreError
from assistai.inference.client import FireworksClient
from assistai.inference.loop import run_turn
from assistai.inference.types import Message
from assistai.manifest import Manifest
from assistai.signal.client import SignalTransport
from assistai.signal.envelopes import InboundText, parse_inbound
from assistai.signal.policy import AccessPolicy
from assistai.signal.ratelimit import RateLimiter
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


class SignalChannel:
    """One inbound text → at most one outbound reply.

    Turns are serialized per sender, so a slow model call for one person does
    not block the other.
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
        self._histories: dict[str, list[Message]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._turns = RateLimiter(settings.signal_rate_limit_per_minute, 60.0)
        # Strangers get a much smaller budget: each reply is outbound traffic
        # from the bot number, which Signal counts against us, not them.
        self._pairing = RateLimiter(settings.signal_pairing_replies_per_hour, 3600.0)
        self._slots = asyncio.Semaphore(settings.max_concurrent_turns)

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
                    continue
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
            await self._converse(inbound, agent)

    async def _converse(self, inbound: InboundText, agent: AgentSpec) -> None:
        if self._fireworks is None or self._manifest is None:
            await self._safe_send(inbound.sender, _NOT_CONFIGURED)
            return
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
        windowed = messages[:]
        apply_window(
            windowed,
            keep=self._settings.history_keep,
            max_age_seconds=self._settings.history_max_age_seconds,
        )
        try:
            self._store.save_history(agent.name, windowed)
        except StoreError:
            log.exception("store.save_failed", agent=agent.name)
            del messages[baseline:]
            await self._safe_send(inbound.sender, _SAVE_FAILED)
            return
        messages[:] = windowed
        await self._safe_send(inbound.sender, reply)

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
