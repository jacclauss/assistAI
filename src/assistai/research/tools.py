"""Model-facing research tools. Results are always untrusted."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import httpx

from assistai.config import Settings
from assistai.errors import HouseholdConfigError, ResearchError
from assistai.inference.client import FireworksClient
from assistai.inference.types import ToolSpec
from assistai.manifest import ModelPin
from assistai.research.fetch import (
    MAX_BYTES,
    MAX_TITLE_CHARS,
    SIDECAR_TIMEOUT_SECONDS,
    FetchedPage,
    fetch_url,
)
from assistai.research.search import SearchHit, web_search
from assistai.research.ssrf import check_url
from assistai.research.summarize import maybe_quarantine

WEB_SEARCH = "web_search"
WEB_FETCH = "web_fetch"

WEB_SEARCH_SPEC = ToolSpec(
    name=WEB_SEARCH,
    description=(
        "Search the public web. Returns titles, URLs, and snippets. "
        "Cite the URLs you use. Follow with web_fetch for a specific page."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, as a person would type it.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

WEB_FETCH_SPEC = ToolSpec(
    name=WEB_FETCH,
    description=(
        "Fetch a public http(s) URL and return extracted markdown. "
        "Cite the URL. Use after web_search when a snippet is not enough."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The http or https URL to fetch.",
            }
        },
        "required": ["url"],
        "additionalProperties": False,
    },
)


def bind_research(
    settings: Settings,
    *,
    fireworks: FireworksClient | None = None,
    quarantine: ModelPin | None = None,
) -> dict[str, Callable[[dict[str, Any]], Awaitable[str]]]:
    """Handlers closed over live settings. Bound when the Signal channel starts."""

    async def search(arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str):
            raise ResearchError("query must be a string")
        hits = await web_search(query, base_url=settings.searxng_base_url)
        return json.dumps(
            {"query": query.strip(), "hits": [_hit_dict(hit) for hit in hits]},
            ensure_ascii=False,
        )

    async def fetch(arguments: dict[str, Any]) -> str:
        url = arguments.get("url")
        if not isinstance(url, str):
            raise ResearchError("url must be a string")
        page = await _fetch_page(url, settings)
        payload = await maybe_quarantine(
            page.markdown,
            url=page.url,
            title=page.title,
            client=fireworks,
            pin=quarantine,
        )
        return json.dumps(payload, ensure_ascii=False)

    return {WEB_SEARCH: search, WEB_FETCH: fetch}


async def _fetch_page(url: str, settings: Settings) -> FetchedPage:
    # Checked here as well as in the sidecar. The sidecar is the enforcing
    # gate, but the gateway should not need to trust it to be the only one.
    check_url(url)
    if settings.extract_base_url:
        return await _fetch_via_sidecar(url, settings.extract_base_url)
    return await fetch_url(url)


# Covers the sidecar's own fetch budget plus extract and the JSON return.
_SIDECAR_TIMEOUT = SIDECAR_TIMEOUT_SECONDS
_SIDECAR_MAX_BYTES = MAX_BYTES + 200_000


async def _fetch_via_sidecar(url: str, base_url: str) -> FetchedPage:
    endpoint = f"{base_url.rstrip('/')}/fetch"
    try:
        async with asyncio.timeout(_SIDECAR_TIMEOUT):
            async with httpx.AsyncClient(
                timeout=_SIDECAR_TIMEOUT, trust_env=False, follow_redirects=False
            ) as http:
                async with http.stream("POST", endpoint, json={"url": url}) as response:
                    raw = await _read_capped(response, max_bytes=_SIDECAR_MAX_BYTES)
                    if response.status_code >= 400:
                        raise ResearchError(_sidecar_error_body(raw))
                    try:
                        payload: object = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ResearchError("extract sidecar returned invalid JSON") from exc
    except TimeoutError as exc:
        raise ResearchError("extract sidecar timed out") from exc
    except httpx.HTTPError as exc:
        raise ResearchError(f"extract sidecar failed ({type(exc).__name__})") from exc
    return _page_from_sidecar(payload)


async def _read_capped(response: httpx.Response, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for part in response.aiter_bytes():
        total += len(part)
        if total > max_bytes:
            raise ResearchError("extract sidecar returned too much")
        chunks.append(part)
    return b"".join(chunks)


def assert_isolated_extract(settings: Settings, tools: Iterable[str]) -> None:
    """Refuse to start if web_fetch would parse hostile HTML in this process.

    trafilatura and lxml are the memory-unsafe part of this phase, which is
    why they belong in a container with no Fireworks key and no state volume.
    An unset sidecar URL silently moves that parser back into the gateway, so
    it fails closed unless the operator says otherwise.
    """
    if settings.extract_base_url or settings.allow_in_process_extract:
        return
    if WEB_FETCH not in set(tools):
        return
    raise HouseholdConfigError(
        "web_fetch is on an agent's tools but ASSISTAI_EXTRACT_BASE_URL is not set, "
        "so HTML would be parsed inside the gateway. Start the extract sidecar "
        "(make up) or set ASSISTAI_ALLOW_IN_PROCESS_EXTRACT=true to accept the risk."
    )


def _page_from_sidecar(payload: object) -> FetchedPage:
    if not isinstance(payload, dict):
        raise ResearchError("extract sidecar returned invalid JSON")
    url = payload.get("url")
    title = payload.get("title")
    markdown = payload.get("markdown")
    if not isinstance(url, str) or not isinstance(markdown, str) or not markdown.strip():
        raise ResearchError("extract sidecar returned no readable text")
    check_url(url)
    if not isinstance(title, str) or not title.strip():
        title = url
    return FetchedPage(
        url=url,
        title=title.strip()[:MAX_TITLE_CHARS] or url,
        markdown=markdown.strip(),
    )


def _sidecar_error_body(raw: bytes) -> str:
    try:
        payload: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return "extract sidecar failed"
    if isinstance(payload, dict):
        message = payload.get("error")
        if isinstance(message, str) and message:
            return message.strip()[:200]
    return "extract sidecar failed"


def _hit_dict(hit: SearchHit) -> dict[str, str]:
    return {"title": hit.title, "url": hit.url, "snippet": hit.snippet}
