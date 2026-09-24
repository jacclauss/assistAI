"""Reject URLs that would let a tool call reach the box or the LAN.

The gateway (and the extract sidecar) must not fetch localhost, link-local
metadata, private ranges, or anything that rebinds to them after a redirect.
Every DNS answer is validated, then the request is dialled at that address
with the original Host and SNI so a second lookup cannot point the TCP
connection at 127.0.0.1. Redirect hops go through the same check again.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import SplitResult, urlsplit

from assistai.errors import ResearchError

MAX_URL_CHARS = 2048

_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
        "internal",
        "local",
        "lan",
        "home",
    }
)
_BLOCKED_SUFFIXES = (".local", ".internal", ".localhost", ".lan", ".home", ".localdomain")
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def check_url(url: str) -> SplitResult:
    """Parse an http(s) URL and reject userinfo, local names, and IP literals."""
    if len(url) > MAX_URL_CHARS:
        raise ResearchError("url is too long")
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"}:
        raise ResearchError("only http and https URLs are allowed")
    if parts.username is not None or parts.password is not None:
        raise ResearchError("URLs with userinfo are not allowed")
    try:
        port = parts.port
    except ValueError as exc:
        raise ResearchError("url has an invalid port") from exc
    if port == 0:
        raise ResearchError("url has an invalid port")
    host = url_host(parts).rstrip(".")
    if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_SUFFIXES):
        raise ResearchError("that host is not allowed")
    literal = _literal_ip(host)
    if literal is not None:
        check_ip(literal)
    return parts


def url_host(parts: SplitResult) -> str:
    """ASCII host for DNS, Host, and SNI. International names go to punycode."""
    host = parts.hostname
    if not host:
        raise ResearchError("url is missing a host")
    if host.isascii():
        return host.lower()
    try:
        return host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ResearchError("url has an invalid host") from exc


def check_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Refuse anything that is not a public unicast address."""
    if ip.version == 6 and ip.ipv4_mapped is not None:
        check_ip(ip.ipv4_mapped)
        return
    if ip.version == 4 and ip in _CGNAT:
        raise ResearchError("that address is not allowed")
    if (
        not ip.is_global
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_private
    ):
        raise ResearchError("that address is not allowed")


async def resolve_public(host: str) -> str:
    """Validate every address ``host`` resolves to and return the one to dial.

    The caller connects to the returned address rather than to the name. A
    second lookup at connect time is what makes DNS rebinding work: the check
    sees a public address and the connection gets 127.0.0.1. Pinning here
    closes that window. Every answer is validated, not just the pinned one, so
    a round-robin record cannot hide a private address behind a public first
    answer.
    """
    literal = _literal_ip(host)
    if literal is not None:
        check_ip(literal)
        return str(literal)
    try:
        infos = await _getaddrinfo(host)
    except (OSError, UnicodeError) as exc:
        # UnicodeError: an empty or over-long label is rejected by the idna codec.
        raise ResearchError("that host could not be resolved") from exc
    pinned: str | None = None
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError as exc:
            raise ResearchError("that host could not be resolved") from exc
        check_ip(ip)
        if pinned is None:
            pinned = str(ip)
    if pinned is None:
        raise ResearchError("that host could not be resolved")
    return pinned


async def _getaddrinfo(host: str) -> list[Any]:
    loop = asyncio.get_running_loop()
    return await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if host.isdigit():
        value = int(host)
        if 0 <= value <= 0xFFFFFFFF:
            return ipaddress.IPv4Address(value)
    return None
