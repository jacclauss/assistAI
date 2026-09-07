"""Parse signal-cli receive payloads into inbound text DMs.

json-rpc mode wraps envelopes several ways (bare, ``{envelope}``, JSON-RPC
``params``). Receipts, typing indicators, stories, and group messages are
ignored: phase 2 is one-to-one text only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from assistai.signal.numbers import InvalidNumberError, normalize_e164


@dataclass(frozen=True)
class InboundText:
    """A one-to-one text message the gateway should consider."""

    sender: str
    text: str
    timestamp: int


def parse_inbound(payload: object) -> InboundText | None:
    """Return a text DM, or ``None`` if the payload is not one."""
    root = _as_dict(payload)
    if root is None:
        return None
    envelope = _envelope_from(root)
    if envelope is None:
        return None
    data = _data_message(envelope)
    if data is None:
        return None
    if data.get("groupInfo"):
        return None
    text = data.get("message")
    if not isinstance(text, str) or not text.strip():
        return None
    sender = envelope.get("sourceNumber") or envelope.get("source")
    if not isinstance(sender, str):
        return None
    try:
        number = normalize_e164(sender)
    except InvalidNumberError:
        return None
    timestamp = envelope.get("timestamp")
    return InboundText(
        sender=number,
        text=text.strip(),
        timestamp=timestamp if isinstance(timestamp, int) else 0,
    )


def _envelope_from(root: dict[str, Any]) -> dict[str, Any] | None:
    envelope = root.get("envelope")
    if isinstance(envelope, dict):
        return envelope
    params = root.get("params")
    if isinstance(params, dict):
        nested = params.get("envelope")
        if isinstance(nested, dict):
            return nested
        result = params.get("result")
        if isinstance(result, dict) and isinstance(result.get("envelope"), dict):
            inner = result["envelope"]
            return inner if isinstance(inner, dict) else None
    if "dataMessage" in root or "source" in root or "sourceNumber" in root:
        return root
    return None


def _data_message(envelope: dict[str, Any]) -> dict[str, Any] | None:
    data = envelope.get("dataMessage")
    if isinstance(data, dict):
        return data
    edit = envelope.get("editMessage")
    if isinstance(edit, dict):
        edited = edit.get("dataMessage")
        if isinstance(edited, dict):
            return edited
    return None


def _as_dict(payload: object) -> dict[str, Any] | None:
    return payload if isinstance(payload, dict) else None
