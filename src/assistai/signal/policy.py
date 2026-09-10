"""Sender allowlist and pairing.

Unknown numbers never reach the model. The default, ``allowlist``, drops them
silently so a stranger never learns a bot lives here. Under ``pairing`` they
get a short-lived code that only an *operator* can approve with
``/approve NNNNNN``.

Operators are exactly the numbers in ``ASSISTAI_SIGNAL_ALLOW_FROM``. Numbers
admitted by pairing can talk to their agent but cannot admit anyone else,
so a single approval does not hand out the power to grant more.

Approvals persist in the state database so a restart does not evict someone
the operator already let in. Pending codes are memory-only and expire.
"""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import structlog

from assistai.store import Store

log = structlog.get_logger(__name__)

APPROVE_RE = re.compile(r"^/(?:approve|pair)\s+(\d{6})$", re.IGNORECASE)
Decision = Literal["allow", "unknown"]

# Bounds the memory a stranger can cause the gateway to allocate.
MAX_PENDING = 64


@dataclass(frozen=True)
class PendingPair:
    sender: str
    code: str
    expires_at: float


class AccessPolicy:
    """Deny-by-default sender gate."""

    def __init__(
        self,
        operators: tuple[str, ...],
        *,
        store: Store,
        pairing_ttl_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._operators = frozenset(operators)
        self._store = store
        self._ttl = pairing_ttl_seconds
        self._clock = clock or _now
        self._approved: set[str] = set(store.approved())
        self._pending: dict[str, PendingPair] = {}

    def allowed(self, sender: str) -> bool:
        return sender in self._operators or sender in self._approved

    def is_operator(self, sender: str) -> bool:
        """Only env-configured numbers may admit new senders."""
        return sender in self._operators

    def decide(self, sender: str) -> Decision:
        return "allow" if self.allowed(sender) else "unknown"

    def request_pair(self, sender: str) -> str:
        """Return the live code for ``sender``, issuing one if needed."""
        now = self._clock()
        self._prune(now)
        existing = self._pending.get(sender)
        if existing is not None and existing.expires_at > now:
            return existing.code
        code = f"{secrets.randbelow(1_000_000):06d}"
        self._pending[sender] = PendingPair(sender, code, now + self._ttl)
        log.info("signal.pairing_issued", sender=sender)
        return code

    def approve(self, code: str, *, approver: str) -> str | None:
        """Admit the pending number for ``code``. Returns the number, or None.

        Raises ``PermissionError`` if ``approver`` is not an operator.
        Raises ``StoreError`` if the number cannot be written; the code stays pending.
        """
        if not self.is_operator(approver):
            raise PermissionError("only an operator may approve pairing codes")
        now = self._clock()
        self._prune(now)
        for sender, pending in list(self._pending.items()):
            if pending.code == code:
                self._store.admit(sender)
                del self._pending[sender]
                self._approved.add(sender)
                log.info("signal.pairing_approved", sender=sender, approver=approver)
                return sender
        return None

    def parse_approve(self, text: str) -> str | None:
        """Return a 6-digit code if ``text`` is solely an approve command."""
        match = APPROVE_RE.fullmatch(text.strip())
        return match.group(1) if match else None

    def _prune(self, now: float) -> None:
        for sender, pending in list(self._pending.items()):
            if pending.expires_at <= now:
                del self._pending[sender]
        while len(self._pending) >= MAX_PENDING:
            oldest = min(self._pending, key=lambda key: self._pending[key].expires_at)
            del self._pending[oldest]
            log.warning("signal.pairing_evicted", sender=oldest)


def _now() -> float:
    return time.time()
