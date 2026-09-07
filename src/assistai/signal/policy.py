"""Sender allowlist and pairing.

Unknown numbers never reach the model. Under ``pairing`` they get a short-lived
code that only an *operator* can approve with ``/approve NNNNNN``. Under
``allowlist`` they are dropped silently.

Operators are exactly the numbers in ``ASSISTAI_SIGNAL_ALLOW_FROM``. Numbers
admitted by pairing can talk to their agent but cannot admit anyone else,
so a single approval does not hand out the power to grant more.

Approvals persist under ``state_dir`` so a restart does not evict someone the
operator already let in.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog

from assistai.signal.numbers import InvalidNumberError, normalize_e164

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
        persist_path: Path,
        pairing_ttl_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._operators = frozenset(operators)
        self._persist_path = persist_path
        self._ttl = pairing_ttl_seconds
        self._clock = clock or _now
        self._approved: set[str] = set()
        self._pending: dict[str, PendingPair] = {}
        self._load()

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
        """
        if not self.is_operator(approver):
            raise PermissionError("only an operator may approve pairing codes")
        now = self._clock()
        self._prune(now)
        for sender, pending in list(self._pending.items()):
            if pending.code == code:
                del self._pending[sender]
                self._approved.add(sender)
                self._save()
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

    def _load(self) -> None:
        if not self._persist_path.is_file():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("signal.allowlist_unreadable", error=type(exc).__name__)
            return
        rows = raw.get("approved") if isinstance(raw, dict) else None
        if not isinstance(rows, list):
            return
        for item in rows:
            if not isinstance(item, str):
                continue
            try:
                self._approved.add(normalize_e164(item))
            except InvalidNumberError:
                continue

    def _save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._persist_path.with_name(self._persist_path.name + ".tmp")
        payload = {"approved": sorted(self._approved)}
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self._persist_path)


def _now() -> float:
    return time.time()
