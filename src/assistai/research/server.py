"""Tiny HTTP sidecar for fetch + extract. No Fireworks key, no store."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from assistai.config import Settings
from assistai.errors import ResearchError
from assistai.research.fetch import (
    SIDECAR_TIMEOUT_SECONDS,
    FetchedPage,
    extract_posted_html,
    fetch_url,
)

log = structlog.get_logger(__name__)
_MAX_BODY = 1_200_000


async def serve(host: str, port: int, settings: Settings) -> None:
    """Listen until cancelled. One request per connection."""
    log.info("extract.listening", host=host, port=port, extract_url=settings.extract_base_url)
    server = await asyncio.start_server(lambda reader, writer: _handle(reader, writer), host, port)
    async with server:
        await server.serve_forever()


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        async with asyncio.timeout(SIDECAR_TIMEOUT_SECONDS):
            status, body = await _dispatch(reader)
        payload = json.dumps(body, ensure_ascii=False).encode()
        headers = (
            f"HTTP/1.1 {status}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(headers.encode() + payload)
        await writer.drain()
    except TimeoutError:
        try:
            writer.write(b"HTTP/1.1 504 Gateway Timeout\r\nConnection: close\r\n\r\n")
            await writer.drain()
        except OSError:
            pass
    except Exception:
        writer.write(b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n")
        try:
            await writer.drain()
        except OSError:
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


async def _dispatch(reader: asyncio.StreamReader) -> tuple[str, dict[str, Any]]:
    try:
        method, path, payload = await _read_request(reader)
        if method == "GET" and path == "/health":
            return "200 OK", {"ok": True}
        if method != "POST":
            return "405 Method Not Allowed", {"error": "method_not_allowed"}
        if path == "/fetch":
            page = await _fetch(payload)
        elif path == "/extract":
            page = extract_posted_html(payload)
        else:
            return "404 Not Found", {"error": "not_found"}
    except ResearchError as exc:
        # A malformed request is the caller's fault, not a server fault.
        return "400 Bad Request", {"error": str(exc)}
    return "200 OK", _page_body(page)


async def _fetch(payload: dict[str, Any]) -> FetchedPage:
    url = payload.get("url")
    if not isinstance(url, str):
        raise ResearchError("url must be a string")
    return await fetch_url(url)


async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, dict[str, Any]]:
    header_blob = await _read_headers(reader)
    lines = header_blob.split("\r\n")
    if not lines or not lines[0]:
        raise ResearchError("invalid request")
    parts = lines[0].split(" ")
    if len(parts) < 2:
        raise ResearchError("invalid request")
    method, path = parts[0].upper(), parts[1].split("?", 1)[0]
    length = 0
    for line in lines[1:]:
        if line.lower().startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError as exc:
                raise ResearchError("invalid content-length") from exc
    if length < 0:
        raise ResearchError("invalid content-length")
    if length > _MAX_BODY:
        raise ResearchError("request is too large")
    body = b""
    if length:
        body = await reader.readexactly(length)
    if method == "GET":
        return method, path, {}
    if not body:
        raise ResearchError("missing body")
    try:
        parsed: object = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchError("body is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ResearchError("body must be a JSON object")
    return method, path, parsed


async def _read_headers(reader: asyncio.StreamReader) -> str:
    buf = bytearray()
    while True:
        line = await reader.readline()
        if not line:
            break
        buf.extend(line)
        if len(buf) > 16_384:
            raise ResearchError("headers too large")
        if buf.endswith(b"\r\n\r\n"):
            break
    return buf.decode("latin-1").rstrip("\r\n")


def _page_body(page: FetchedPage) -> dict[str, str]:
    return {"url": page.url, "title": page.title, "markdown": page.markdown}
