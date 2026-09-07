"""Server-sent event parsing for OpenAI-compatible chat streams.

Fireworks sends ``data: {json}`` frames and a terminal ``data: [DONE]``.
TCP chunks can split a frame mid-line, so callers feed raw text incrementally.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


class SSEBuffer:
    """Accumulate streamed text and yield complete ``data`` payloads."""

    def __init__(self) -> None:
        self._pending = ""

    def push(self, chunk: str) -> Iterator[str]:
        """Yield complete data payloads from ``chunk``, holding a partial line."""
        self._pending += chunk
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            payload = _payload_from_line(line)
            if payload is not None:
                yield payload

    def flush(self) -> Iterator[str]:
        """Yield a trailing payload if the stream ended without a newline."""
        if not self._pending:
            return
        payload = _payload_from_line(self._pending)
        self._pending = ""
        if payload is not None:
            yield payload


def _payload_from_line(line: str) -> str | None:
    stripped = line.rstrip("\r")
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    return stripped[5:].lstrip()


DONE = "[DONE]"


def parse_sse_json(payload: str) -> dict[str, Any] | None:
    """Decode a data payload. ``[DONE]`` and empty frames become ``None``."""
    if payload == DONE or payload == "":
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid SSE JSON: {payload[:200]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("SSE JSON must be an object")
    return parsed
