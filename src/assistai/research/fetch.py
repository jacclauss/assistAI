"""HTTP GET plus readability extract. SSRF-hardened, byte- and time-capped."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import SplitResult, urljoin, urlunsplit

import httpx
import trafilatura

from assistai.errors import ResearchError
from assistai.research.ssrf import check_url, resolve_public

MAX_BYTES = 1_000_000
MAX_REDIRECTS = 5
DEFAULT_TIMEOUT = 15.0
# A whole fetch, not one socket operation. Five hops of per-operation timeouts
# would otherwise hold a turn open for minutes.
MAX_TOTAL_SECONDS = 30.0
MAX_TITLE_CHARS = 200
# Sidecar needs the fetch budget plus time to parse and JSON-encode the page.
SIDECAR_TIMEOUT_SECONDS = MAX_TOTAL_SECONDS + 15.0
_ACCEPT = "text/html,application/xhtml+xml,text/plain;q=0.9"
_USER_AGENT = "AssistAI/0.0"
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})

# Returns the validated address to connect to, so the name is never resolved
# a second time between the check and the connection.
Resolver = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class FetchedPage:
    """Extracted page the model may cite. Always untrusted at the broker."""

    url: str
    title: str
    markdown: str


async def fetch_url(
    url: str,
    *,
    http: httpx.AsyncClient | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT,
    total_seconds: float = MAX_TOTAL_SECONDS,
    max_bytes: int = MAX_BYTES,
    resolve: Resolver = resolve_public,
) -> FetchedPage:
    """GET ``url`` after SSRF checks and extract markdown."""
    html, final_url = await get_html(
        url,
        http=http,
        timeout_seconds=timeout_seconds,
        total_seconds=total_seconds,
        max_bytes=max_bytes,
        resolve=resolve,
    )
    return extract_page(html, url=final_url)


async def get_html(
    url: str,
    *,
    http: httpx.AsyncClient | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT,
    total_seconds: float = MAX_TOTAL_SECONDS,
    max_bytes: int = MAX_BYTES,
    resolve: Resolver = resolve_public,
) -> tuple[str, str]:
    """Return ``(html, final_url)``. Redirects are re-validated at every hop."""
    owns = http is None
    # Keepalive is off: pooling is keyed on the pinned IP, so a redirect from
    # a.cdn.com to b.cdn.com that shares an address would reuse a TLS session
    # whose SNI is still a.
    client = http or httpx.AsyncClient(
        timeout=timeout_seconds,
        follow_redirects=False,
        trust_env=False,
        limits=httpx.Limits(max_keepalive_connections=0),
    )
    try:
        async with asyncio.timeout(total_seconds):
            return await _walk(client, url, max_bytes=max_bytes, resolve=resolve)
    except TimeoutError as exc:
        raise ResearchError("that fetch took too long") from exc
    finally:
        if owns:
            await client.aclose()


async def _walk(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int,
    resolve: Resolver,
) -> tuple[str, str]:
    """Follow redirects by hand so each hop is checked before it is dialled."""
    current = url
    for _ in range(MAX_REDIRECTS):
        parts = check_url(current)
        host = parts.hostname
        if host is None:
            raise ResearchError("url is missing a host")
        address = await resolve(host)
        location: str | None = None
        try:
            # Streamed, not buffered: a non-streaming get reads the whole body
            # before max_bytes could ever apply, so a gzip bomb would be fully
            # decompressed into memory first.
            async with client.stream(
                "GET",
                _pinned_url(parts, address),
                headers=_headers(parts),
                extensions=_extensions(parts, host),
            ) as response:
                if response.has_redirect_location:
                    location = response.headers.get("location")
                    if not isinstance(location, str) or not location:
                        raise ResearchError("redirect is missing a location")
                elif 300 <= response.status_code < 400:
                    raise ResearchError("redirect is missing a location")
                else:
                    if response.status_code >= 400:
                        raise ResearchError(f"page returned HTTP {response.status_code}")
                    if _declared_unreadable(response):
                        raise ResearchError("unsupported content type")
                    html = await _read_body(response, max_bytes=max_bytes)
                    if not _looks_extractable(response, html):
                        raise ResearchError("unsupported content type")
                    # The real URL, not the pinned address: this is what gets cited.
                    return html, current
        except httpx.HTTPError as exc:
            raise ResearchError(f"fetch failed ({type(exc).__name__})") from exc
        current = urljoin(current, location)
    raise ResearchError("too many redirects")


def _pinned_url(parts: SplitResult, address: str) -> str:
    """The same request, addressed to the validated IP instead of the name."""
    literal = f"[{address}]" if ":" in address else address
    netloc = f"{literal}:{parts.port}" if parts.port else literal
    return urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))


def _headers(parts: SplitResult) -> dict[str, str]:
    return {
        "User-Agent": _USER_AGENT,
        "Accept": _ACCEPT,
        # Dialling an IP would otherwise send the IP as Host and miss vhosts.
        "Host": _host_header(parts),
    }


def _host_header(parts: SplitResult) -> str:
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{parts.port}" if parts.port else host


def _extensions(parts: SplitResult, host: str) -> dict[str, Any]:
    """Keep TLS pointed at the name so SNI and certificate checks still apply."""
    if parts.scheme != "https":
        return {}
    return {"sni_hostname": host}


def extract_page(html: str, *, url: str) -> FetchedPage:
    """Turn HTML into markdown. Empty extract is a hard failure."""
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_links=True,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if not markdown or not markdown.strip():
        raise ResearchError("no readable text on that page")
    title = _title(html) or url
    return FetchedPage(
        url=url, title=title.strip()[:MAX_TITLE_CHARS] or url, markdown=markdown.strip()
    )


def extract_posted_html(arguments: dict[str, Any]) -> FetchedPage:
    """Sidecar ``/extract`` body: HTML the caller already fetched."""
    html = arguments.get("html")
    url = arguments.get("url")
    if not isinstance(html, str) or not html.strip():
        raise ResearchError("html must be a non-empty string")
    if not isinstance(url, str) or not url.strip():
        raise ResearchError("url must be a string")
    return extract_page(html, url=url)


def _title(html: str) -> str:
    metadata = trafilatura.extract_metadata(html)
    if metadata is not None and metadata.title:
        return metadata.title.strip()
    return ""


async def _read_body(
    response: httpx.Response,
    *,
    max_bytes: int,
    too_large: str = "that page is too large",
    failed: str = "fetch failed",
) -> str:
    """Read up to ``max_bytes`` of decoded body, then give up on the page.

    httpx decompresses as it iterates, so the cap applies to the expanded
    size and a small compressed bomb cannot outgrow it.
    """
    chunks: list[bytes] = []
    total = 0
    try:
        async for part in response.aiter_bytes():
            total += len(part)
            if total > max_bytes:
                raise ResearchError(too_large)
            chunks.append(part)
    except httpx.HTTPError as exc:
        raise ResearchError(f"{failed} ({type(exc).__name__})") from exc
    return b"".join(chunks).decode("utf-8", errors="replace")


def _media_type(response: httpx.Response) -> str:
    header = response.headers.get("content-type")
    if not isinstance(header, str):
        return ""
    return header.split(";", 1)[0].strip().lower()


def _declared_unreadable(response: httpx.Response) -> bool:
    """Reject known-bad types before the body is pulled into memory."""
    media = _media_type(response)
    return bool(media) and media not in _HTML_TYPES


def _looks_extractable(response: httpx.Response, html: str) -> bool:
    media = _media_type(response)
    if media in _HTML_TYPES:
        return True
    if media:
        return False
    lowered = html.lstrip()[:200].lower()
    return lowered.startswith("<!doctype html") or lowered.startswith("<html")
