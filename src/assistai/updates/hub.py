"""Read published tags from Docker Hub. No other host is contacted."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

import httpx

from assistai.errors import UpdateError
from assistai.updates.check import PublishedTag, TagPage

_HUB = "https://hub.docker.com"
_REPO = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*/[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_PAGE_CAP = 5
_BODY_CAP = 1_000_000


class DockerHubTags:
    """Tag list for one Docker Hub repository. The client is not a pull."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns = client is None

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()

    async def tags(self, repository: str) -> TagPage:
        if _REPO.fullmatch(repository) is None:
            raise UpdateError("image repository is not a Docker Hub name")
        url = f"{_HUB}/v2/repositories/{repository}/tags?page_size=100"
        found: list[PublishedTag] = []
        complete = True
        for _page in range(_PAGE_CAP):
            payload = await self._get_json(url)
            results = payload.get("results")
            if not isinstance(results, list):
                raise UpdateError("Docker Hub returned an unexpected tag list")
            for item in results:
                tag = _tag(item)
                if tag is not None:
                    found.append(tag)
            nxt = payload.get("next")
            if not isinstance(nxt, str) or not nxt:
                break
            url = _next_page(nxt, repository)
        else:
            complete = False
        return TagPage(tags=tuple(found), complete=complete)

    async def _get_json(self, url: str) -> dict[str, Any]:
        try:
            async with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise UpdateError("Docker Hub refused the tag list")
                body = await _read_capped(response)
        except httpx.HTTPError as exc:
            raise UpdateError("Docker Hub is unreachable") from exc
        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise UpdateError("Docker Hub returned an unexpected tag list") from exc
        if not isinstance(payload, dict):
            raise UpdateError("Docker Hub returned an unexpected tag list")
        return payload


async def _read_capped(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _BODY_CAP:
            raise UpdateError("Docker Hub returned an unexpected tag list")
        chunks.append(chunk)
    return b"".join(chunks)


def _next_page(url: str, repository: str) -> str:
    """Stay on the same Docker Hub tag list. A next link must not change host or repository."""
    parts = urlsplit(url)
    expected = f"/v2/repositories/{repository}/tags"
    if parts.scheme != "https" or parts.hostname != "hub.docker.com":
        raise UpdateError("Docker Hub returned a tag page on another host")
    if parts.port not in (None, 443) or parts.username or parts.password or parts.path != expected:
        raise UpdateError("Docker Hub returned an unexpected tag page")
    return url


def _tag(item: object) -> PublishedTag | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    if not isinstance(name, str) or not name:
        return None
    digest = item.get("digest")
    if digest is None:
        digest = ""
    if not isinstance(digest, str):
        return None
    return PublishedTag(name=name, digest=digest)
