"""iCloud CalDAV for one named calendar. Writes are a single new event."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, date, datetime, time, timedelta
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

import defusedxml.ElementTree as ElementTree
import httpx
import structlog

from assistai.bounded import ACCEPT_ENCODING, BodyTooLargeError, BodyUnreadableError, read_bounded
from assistai.calendar.draft import CalendarDraft
from assistai.calendar.ics import render_event
from assistai.calendar.parse import CalendarEvent, events_on
from assistai.config import Settings
from assistai.errors import CalendarError, ResearchError
from assistai.research.ssrf import check_url

log = structlog.get_logger(__name__)

MAX_EVENTS = 40
MAX_RESPONSE_BYTES = 2_000_000
MAX_REDIRECTS = 5
DEFAULT_TIMEOUT = 20.0
TOTAL_SECONDS = 30.0

_PROPFIND_PRINCIPAL = """<?xml version="1.0" encoding="UTF-8"?>
<d:propfind xmlns:d="DAV:">
  <d:prop><d:current-user-principal/></d:prop>
</d:propfind>
"""

_PROPFIND_HOME = """<?xml version="1.0" encoding="UTF-8"?>
<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:prop><c:calendar-home-set/></d:prop>
</d:propfind>
"""

_PROPFIND_CALENDARS = """<?xml version="1.0" encoding="UTF-8"?>
<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:prop>
    <d:displayname/>
    <d:resourcetype/>
  </d:prop>
</d:propfind>
"""


class Agenda:
    """One local day on the configured calendar."""

    def __init__(
        self,
        *,
        calendar: str,
        timezone: str,
        day: date,
        events: list[CalendarEvent],
        truncated: bool,
    ) -> None:
        self.calendar = calendar
        self.timezone = timezone
        self.day = day
        self.events = events
        self.truncated = truncated


class CalendarClient:
    """Discovers the named calendar once, then queries a single local day."""

    def __init__(self, settings: Settings, *, http: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._http = http
        self._calendar_url: str | None = None

    async def agenda(self, day: date | None = None, *, now: datetime | None = None) -> Agenda:
        """Events overlapping ``day``. Omitted day means today in the household zone."""
        _require_configured(self._settings)
        zone = ZoneInfo(self._settings.timezone)
        moment = now if now is not None else datetime.now(zone)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=zone)
        if day is None:
            day = moment.astimezone(zone).date()
        start = datetime.combine(day, time.min, tzinfo=zone)
        end = start + timedelta(days=1)
        base_host = _configured_host(self._settings.caldav_base_url)
        owns = self._http is None
        http = self._http or httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=False,
            trust_env=False,
            auth=_auth(self._settings),
        )
        try:
            async with asyncio.timeout(TOTAL_SECONDS):
                url = await self._collection(http, base_host)
                events = await self._query(
                    http, url, base_host, start=start, end=end, day=day, zone=zone
                )
        except TimeoutError as exc:
            raise CalendarError("that calendar took too long") from exc
        finally:
            if owns:
                await http.aclose()
        clipped = events[:MAX_EVENTS]
        return Agenda(
            calendar=self._settings.caldav_calendar.strip(),
            timezone=self._settings.timezone,
            day=day,
            events=clipped,
            truncated=len(events) > MAX_EVENTS,
        )

    async def create(self, draft: CalendarDraft, *, stamp: datetime | None = None) -> str:
        """PUT one new event on the configured calendar. Returns a short confirmation."""
        _require_configured(self._settings)
        zone = ZoneInfo(self._settings.timezone)
        moment = stamp if stamp is not None else datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        base_host = _configured_host(self._settings.caldav_base_url)
        owns = self._http is None
        http = self._http or httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=False,
            trust_env=False,
            auth=_auth(self._settings),
        )
        try:
            async with asyncio.timeout(TOTAL_SECONDS):
                collection = await self._collection(http, base_host)
                await self._put_new(http, collection, base_host, draft, zone, moment)
        except TimeoutError as exc:
            raise CalendarError("that calendar took too long") from exc
        finally:
            if owns:
                await http.aclose()
        return f"Added {draft.summary!r} for {draft.label()}."

    async def _collection(self, http: httpx.AsyncClient, base_host: str) -> str:
        if self._calendar_url is not None:
            return self._calendar_url
        principal = await self._propfind(
            http,
            self._settings.caldav_base_url,
            base_host,
            body=_PROPFIND_PRINCIPAL,
            depth="0",
        )
        principal_href = _href_under(principal, "current-user-principal")
        if principal_href is None:
            raise CalendarError("calendar is unavailable")
        home_doc = await self._propfind(
            http,
            _join(self._settings.caldav_base_url, principal_href),
            base_host,
            body=_PROPFIND_HOME,
            depth="0",
        )
        home_href = _href_under(home_doc, "calendar-home-set")
        if home_href is None:
            raise CalendarError("calendar is unavailable")
        listed = await self._propfind(
            http,
            _join(self._settings.caldav_base_url, home_href),
            base_host,
            body=_PROPFIND_CALENDARS,
            depth="1",
        )
        url = _match_calendar(listed, self._settings.caldav_calendar)
        self._calendar_url = _join(self._settings.caldav_base_url, url)
        guard_url(self._calendar_url, base_host)
        return self._calendar_url

    async def _query(
        self,
        http: httpx.AsyncClient,
        url: str,
        base_host: str,
        *,
        start: datetime,
        end: datetime,
        day: date,
        zone: ZoneInfo,
    ) -> list[CalendarEvent]:
        window_start = _stamp(start)
        window_end = _stamp(end)
        body = _report_body(window_start, window_end)
        status, raw = await self._send(http, "REPORT", url, base_host, body=body, depth="1")
        if status >= 400:
            self._calendar_url = None
        _raise_for_status(status)
        try:
            root = _xml(raw)
        except CalendarError:
            self._calendar_url = None
            raise
        found: list[CalendarEvent] = []
        for chunk in _calendar_data(root):
            found.extend(events_on(chunk, day=day, zone=zone))
        found.sort(key=lambda item: (item.start, item.summary))
        return found

    async def _propfind(
        self,
        http: httpx.AsyncClient,
        url: str,
        base_host: str,
        *,
        body: str,
        depth: str,
    ) -> ElementTree.Element:
        status, raw = await self._send(http, "PROPFIND", url, base_host, body=body, depth=depth)
        _raise_for_status(status)
        return _xml(raw)

    async def _put_new(
        self,
        http: httpx.AsyncClient,
        collection: str,
        base_host: str,
        draft: CalendarDraft,
        zone: ZoneInfo,
        stamp: datetime,
    ) -> None:
        folder = collection if collection.endswith("/") else collection + "/"
        for _attempt in range(2):
            uid = uuid.uuid4().hex
            body = render_event(draft, uid=uid, zone=zone, stamp=stamp)
            target = urljoin(folder, f"{uid}.ics")
            status, _raw = await self._send(
                http,
                "PUT",
                target,
                base_host,
                body=body,
                depth=None,
                content_type="text/calendar; charset=utf-8",
                if_none_match=True,
            )
            if status in {200, 201, 204}:
                log.info("calendar.created")
                return
            if status not in {409, 412}:
                _raise_for_status(status)
        raise CalendarError("calendar could not store that event")

    async def _send(
        self,
        http: httpx.AsyncClient,
        method: str,
        url: str,
        base_host: str,
        *,
        body: str,
        depth: str | None,
        content_type: str = "application/xml; charset=utf-8",
        if_none_match: bool = False,
    ) -> tuple[int, bytes]:
        current = url
        payload = body.encode("utf-8")
        headers = {
            "Content-Type": content_type,
            "Accept-Encoding": ACCEPT_ENCODING,
        }
        if depth is not None:
            headers["Depth"] = depth
        if if_none_match:
            headers["If-None-Match"] = "*"
        for _ in range(MAX_REDIRECTS + 1):
            guard_url(current, base_host)
            try:
                async with http.stream(
                    method,
                    current,
                    headers=headers,
                    content=payload,
                    auth=_auth(self._settings),
                ) as response:
                    status = response.status_code
                    if status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not isinstance(location, str) or not location.strip():
                            raise CalendarError("calendar is unavailable")
                        current = urljoin(current, location.strip())
                        continue
                    raw = await _read_capped(response)
            except httpx.HTTPError as exc:
                log.warning("calendar.http_failed", error=type(exc).__name__)
                raise CalendarError("calendar is unavailable") from exc
            return status, raw
        raise CalendarError("calendar is unavailable")


def guard_url(url: str, configured_host: str) -> None:
    """Refuse cleartext, private hosts, and anything outside the calendar server."""
    try:
        parts = check_url(url)
    except ResearchError as exc:
        raise CalendarError("calendar server returned a URL that is not allowed") from exc
    if parts.scheme != "https":
        raise CalendarError("calendar server must be https")
    host = parts.hostname or ""
    if not host_allowed(host, configured_host):
        raise CalendarError("calendar server returned a URL that is not allowed")


def host_allowed(host: str, configured: str) -> bool:
    """iCloud answers from ``pXX-caldav.icloud.com`` after discovery on ``caldav.icloud.com``."""
    cleaned = host.rstrip(".").lower()
    base = configured.rstrip(".").lower()
    if cleaned == base:
        return True
    if base == "icloud.com" or base.endswith(".icloud.com"):
        return cleaned == "icloud.com" or cleaned.endswith(".icloud.com")
    return False


def _require_configured(settings: Settings) -> None:
    password = settings.caldav_password
    if (
        not settings.caldav_username.strip()
        or password is None
        or not password.get_secret_value()
        or not settings.caldav_calendar.strip()
    ):
        raise CalendarError("calendar is not configured")


def _auth(settings: Settings) -> httpx.BasicAuth:
    password = settings.caldav_password
    secret = password.get_secret_value() if password is not None else ""
    return httpx.BasicAuth(settings.caldav_username, secret)


def _configured_host(url: str) -> str:
    host = urlsplit(url).hostname
    if host is None:
        raise CalendarError("calendar server must be https")
    return host


def _join(base: str, href: str) -> str:
    return urljoin(base if base.endswith("/") else base + "/", href)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _report_body(start: str, end: str) -> str:
    # expand asks iCloud to turn RRULEs into instances inside this window.
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:prop>
    <c:calendar-data>
      <c:expand start="{start}" end="{end}"/>
    </c:calendar-data>
  </d:prop>
  <c:filter>
    <c:comp-filter name="VCALENDAR">
      <c:comp-filter name="VEVENT">
        <c:time-range start="{start}" end="{end}"/>
      </c:comp-filter>
    </c:comp-filter>
  </c:filter>
</c:calendar-query>
"""


def _raise_for_status(status: int) -> None:
    if status in {401, 403}:
        raise CalendarError("calendar login was rejected")
    if status >= 400 or status < 200:
        raise CalendarError("calendar is unavailable")


def _xml(raw: bytes) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise CalendarError("calendar returned unreadable data") from exc


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _href_under(root: ElementTree.Element, parent: str) -> str | None:
    for node in root.iter():
        if _local(node.tag) != parent:
            continue
        for child in node.iter():
            if child is node:
                continue
            text = child.text
            if _local(child.tag) == "href" and isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _match_calendar(root: ElementTree.Element, calendar_name: str) -> str:
    wanted = calendar_name.strip().casefold()
    matches: list[str] = []
    for response in list(root):
        if _local(response.tag) != "response":
            continue
        href: str | None = None
        name: str | None = None
        is_calendar = False
        for node in response.iter():
            local = _local(node.tag)
            if local == "href" and href is None and (node.text or "").strip():
                href = node.text.strip()
            elif local == "displayname" and (node.text or "").strip():
                name = node.text.strip()
            elif local == "calendar":
                is_calendar = True
        if not is_calendar or href is None or name is None:
            continue
        if name.casefold() == wanted:
            matches.append(href)
    if not matches:
        shown = calendar_name.strip()[:100]
        raise CalendarError(f"no calendar named {shown}")
    if len(matches) > 1:
        raise CalendarError("that calendar name matches more than one calendar")
    return matches[0]


def _calendar_data(root: ElementTree.Element) -> list[str]:
    chunks: list[str] = []
    for node in root.iter():
        if _local(node.tag) == "calendar-data" and (node.text or "").strip():
            chunks.append(node.text or "")
    return chunks


async def _read_capped(response: httpx.Response) -> bytes:
    try:
        return await read_bounded(response, max_bytes=MAX_RESPONSE_BYTES)
    except BodyTooLargeError:
        raise CalendarError("calendar returned too much") from None
    except BodyUnreadableError:
        raise CalendarError("calendar returned unreadable data") from None
