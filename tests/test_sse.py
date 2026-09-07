from __future__ import annotations

import pytest

from assistai.inference.sse import DONE, SSEBuffer, parse_sse_json


def test_split_across_chunks() -> None:
    buffer = SSEBuffer()

    first = list(buffer.push('data: {"a":'))
    second = list(buffer.push("1}\n"))

    assert first == []
    assert second == ['{"a":1}']


def test_ignores_comments_and_blank_lines() -> None:
    buffer = SSEBuffer()
    payloads = list(buffer.push(": ping\n\ndata: hi\n\n"))

    assert payloads == ["hi"]


def test_crlf() -> None:
    buffer = SSEBuffer()
    payloads = list(buffer.push("data: x\r\n"))

    assert payloads == ["x"]


def test_flush_trailing_payload() -> None:
    buffer = SSEBuffer()
    assert list(buffer.push("data: leftover")) == []
    assert list(buffer.flush()) == ["leftover"]


def test_done_and_empty_are_none() -> None:
    assert parse_sse_json(DONE) is None
    assert parse_sse_json("") is None
    assert parse_sse_json('{"ok": true}') == {"ok": True}


def test_invalid_json_raises() -> None:
    with pytest.raises(ValueError, match="invalid SSE JSON"):
        parse_sse_json("{nope")


def test_non_object_json_rejected() -> None:
    with pytest.raises(ValueError, match="object"):
        parse_sse_json("[1]")
