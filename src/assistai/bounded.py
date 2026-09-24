"""Read an HTTP body with a hard cap on the decompressed size.

httpx decompresses a whole network chunk at once and applies every layer of a
stacked ``Content-Encoding`` such as ``gzip, gzip``, so counting bytes after
``aiter_bytes`` still lets a few hundred bytes expand to hundreds of MB before
the count is checked. This reads the raw bytes and inflates them with an
output limit, and it accepts at most one encoding.
"""

from __future__ import annotations

import zlib

import httpx

# Sent on every capped request so the server does not pick br or zstd, which
# this reader does not inflate.
ACCEPT_ENCODING = "gzip, deflate"


class BodyTooLargeError(Exception):
    """The decoded body would exceed the cap."""


class BodyUnreadableError(Exception):
    """Unsupported, stacked, or corrupt content encoding."""


class _Inflater:
    def __init__(self, coding: str) -> None:
        self._coding = coding
        self._raw_fallback = coding == "deflate"
        self._started = False
        self._obj = zlib.decompressobj(_wbits(coding))

    def feed(self, data: bytes, limit: int) -> bytes:
        try:
            out = self._obj.decompress(data, limit)
        except zlib.error:
            # Some servers send raw deflate without the zlib header.
            if self._raw_fallback and not self._started:
                self._raw_fallback = False
                self._obj = zlib.decompressobj(-zlib.MAX_WBITS)
                return self.feed(data, limit)
            raise BodyUnreadableError("corrupt content encoding") from None
        self._started = True
        return out

    @property
    def tail(self) -> bytes:
        return self._obj.unconsumed_tail

    def finish(self, limit: int) -> bytes:
        try:
            out = self._obj.flush()
        except zlib.error:
            raise BodyUnreadableError("corrupt content encoding") from None
        if len(out) > limit:
            raise BodyTooLargeError
        return out


def _wbits(coding: str) -> int:
    if coding in {"gzip", "x-gzip"}:
        return 16 + zlib.MAX_WBITS
    return zlib.MAX_WBITS


def _inflater(response: httpx.Response) -> _Inflater | None:
    header = response.headers.get("content-encoding") or ""
    codings = [
        part.strip().lower()
        for part in header.split(",")
        if part.strip() and part.strip().lower() != "identity"
    ]
    if not codings:
        return None
    if len(codings) > 1:
        raise BodyUnreadableError("stacked content encodings are not supported")
    coding = codings[0]
    if coding not in {"gzip", "x-gzip", "deflate"}:
        raise BodyUnreadableError("unsupported content encoding")
    return _Inflater(coding)


async def read_bounded(response: httpx.Response, *, max_bytes: int) -> bytes:
    """Decoded body, or ``BodyTooLargeError`` once it would pass ``max_bytes``.

    ``httpx.HTTPError`` from the stream propagates so callers can word it.
    """
    if response.is_stream_consumed:
        # Built from in-memory bytes (a test double); httpx already decoded it.
        content = response.content
        if len(content) > max_bytes:
            raise BodyTooLargeError
        return content
    inflater = _inflater(response)
    chunks: list[bytes] = []
    total = 0
    async for raw in response.aiter_raw():
        if inflater is None:
            total += len(raw)
            if total > max_bytes:
                raise BodyTooLargeError
            chunks.append(raw)
            continue
        data = raw
        while data:
            # One byte past the cap is enough to know the page is too large.
            out = inflater.feed(data, max_bytes - total + 1)
            total += len(out)
            if total > max_bytes:
                raise BodyTooLargeError
            chunks.append(out)
            data = inflater.tail
    if inflater is not None:
        chunks.append(inflater.finish(max_bytes - total))
    return b"".join(chunks)
