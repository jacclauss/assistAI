"""Parse signal-cli receive payloads into inbound DMs.

json-rpc mode wraps envelopes several ways (bare, ``{envelope}``, JSON-RPC
``params``). Receipts, typing indicators, stories, and group messages are
ignored. Attachments are noted by name and type; the binary is not kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from assistai.signal.numbers import InvalidNumberError, is_uuid, normalize_e164


@dataclass(frozen=True)
class InboundAttachment:
    """Identity of an inbound file. The binary is not kept.

    Phase 6 records that something arrived so a later extractor can fill in
    text. Empty ``extracted`` means the file was noted, not summarized.
    """

    name: str
    content_type: str
    size: int
    extracted: str = ""


@dataclass(frozen=True)
class InboundText:
    """A one-to-one text message the gateway should consider."""

    sender: str
    text: str
    timestamp: int
    attachments: tuple[InboundAttachment, ...] = field(default_factory=tuple)


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
    attachments = _attachments(data)
    text = data.get("message")
    if not isinstance(text, str) or not text.strip():
        if not attachments:
            return None
        text = ""
    sender = _sender_of(envelope)
    if sender is None:
        return None
    timestamp = envelope.get("timestamp")
    return InboundText(
        sender=sender,
        text=text.strip(),
        timestamp=timestamp if isinstance(timestamp, int) else 0,
        attachments=attachments,
    )


def _sender_of(envelope: dict[str, Any]) -> str | None:
    """Prefer E.164. Phone-number privacy often leaves only a UUID."""
    e164: str | None = None
    uuid: str | None = None
    for raw in (
        envelope.get("sourceNumber"),
        envelope.get("sourceUuid"),
        envelope.get("source"),
    ):
        if not isinstance(raw, str):
            continue
        value = raw.strip()
        if not value:
            continue
        try:
            e164 = e164 or normalize_e164(value)
        except InvalidNumberError:
            if uuid is None and is_uuid(value):
                uuid = value.lower()
    return e164 or uuid


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


def _attachments(data: dict[str, Any]) -> tuple[InboundAttachment, ...]:
    raw = data.get("attachments")
    if not isinstance(raw, list):
        return ()
    found: list[InboundAttachment] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("filename") or item.get("id") or "attachment"
        content_type = (
            item.get("contentType") or item.get("content_type") or "application/octet-stream"
        )
        size = item.get("size") or 0
        if not isinstance(name, str) or not name:
            name = "attachment"
        if not isinstance(content_type, str) or not content_type:
            content_type = "application/octet-stream"
        if not isinstance(size, int) or size < 0:
            size = 0
        found.append(InboundAttachment(name=name, content_type=content_type, size=size))
    return tuple(found)
