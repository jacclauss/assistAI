"""SearXNG-backed web search. Titles, URLs, and snippets only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from assistai.bounded import ACCEPT_ENCODING
from assistai.errors import ResearchError
from assistai.research.fetch import _read_body
from assistai.research.ssrf import check_url

MAX_QUERY_CHARS = 200
MAX_HITS = 8
DEFAULT_TIMEOUT = 15.0
MAX_RESPONSE_BYTES = 512_000
MAX_TITLE_CHARS = 200
MAX_SNIPPET_CHARS = 400


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


async def web_search(
    query: str,
    *,
    base_url: str,
    http: httpx.AsyncClient | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT,
    limit: int = MAX_HITS,
) -> list[SearchHit]:
    """Query SearXNG and keep hits whose URLs would pass the SSRF URL check."""
    cleaned = query.strip()
    if not cleaned:
        raise ResearchError("query is empty")
    if len(cleaned) > MAX_QUERY_CHARS:
        raise ResearchError("query is too long")
    owns = http is None
    client = http or httpx.AsyncClient(
        timeout=timeout_seconds, trust_env=False, follow_redirects=False
    )
    url = f"{base_url.rstrip('/')}/search"
    try:
        try:
            async with client.stream(
                "GET",
                url,
                params={"q": cleaned, "format": "json", "language": "en"},
                headers={"Accept-Encoding": ACCEPT_ENCODING},
            ) as response:
                if response.status_code >= 400:
                    raise ResearchError("search is unavailable")
                raw = await _read_body(
                    response,
                    max_bytes=MAX_RESPONSE_BYTES,
                    too_large="search returned too much",
                    failed="search failed",
                )
        except httpx.HTTPError as exc:
            raise ResearchError(f"search failed ({type(exc).__name__})") from exc
        try:
            payload: object = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResearchError("search returned invalid JSON") from exc
    finally:
        if owns:
            await client.aclose()
    return _hits(payload, limit=limit)


def _hits(payload: object, *, limit: int) -> list[SearchHit]:
    if not isinstance(payload, dict):
        raise ResearchError("search returned invalid JSON")
    rows = payload.get("results")
    if not isinstance(rows, list):
        return []
    hits: list[SearchHit] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        hit = _hit(row)
        if hit is None:
            continue
        hits.append(hit)
        if len(hits) >= limit:
            break
    return hits


def _hit(row: dict[str, Any]) -> SearchHit | None:
    url = row.get("url")
    title = row.get("title")
    snippet = row.get("content") or row.get("snippet") or ""
    if not isinstance(url, str) or not url.strip():
        return None
    if not isinstance(title, str):
        title = url
    if not isinstance(snippet, str):
        snippet = ""
    try:
        check_url(url)
    except ResearchError:
        return None
    return SearchHit(
        title=(title.strip() or url)[:MAX_TITLE_CHARS],
        url=url.strip(),
        snippet=snippet.strip()[:MAX_SNIPPET_CHARS],
    )
