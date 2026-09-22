from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest

from assistai.broker import ToolBroker, builtin_catalog
from assistai.calendar.client import CalendarClient, host_allowed
from assistai.calendar.parse import events_on
from assistai.calendar.tools import CALENDAR_TODAY, bind_calendar
from assistai.config import Settings
from assistai.errors import CalendarError
from tests.agent_fakes import agent, household
from tests.fakes import settings, tool_call

_PASSWORD = "app-secret-password"
_ICS = """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:dentist
DTSTART:20260922T140000Z
DTEND:20260922T150000Z
SUMMARY:Dentist
LOCATION:Office
END:VEVENT
BEGIN:VEVENT
UID:skip
STATUS:CANCELLED
DTSTART:20260922T160000Z
DTEND:20260922T170000Z
SUMMARY:Skip me
END:VEVENT
BEGIN:VEVENT
UID:trip
DTSTART;VALUE=DATE:20260922
DTEND;VALUE=DATE:20260923
SUMMARY:Trip
END:VEVENT
BEGIN:VEVENT
UID:tomorrow
DTSTART:20260923T060000Z
DTEND:20260923T070000Z
SUMMARY:Already tomorrow
END:VEVENT
END:VCALENDAR
"""


def _configured(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "caldav_base_url": "https://caldav.test/",
        "caldav_username": "ada@icloud.com",
        "caldav_password": _PASSWORD,
        "caldav_calendar": "Home",
        "timezone": "America/Chicago",
    }
    values.update(overrides)
    return settings(**values)


def _xml(body: str, status: int = 207) -> httpx.Response:
    return httpx.Response(status, content=body.encode())


def _principal() -> httpx.Response:
    return _xml(
        """<?xml version="1.0"?>
        <d:multistatus xmlns:d="DAV:">
          <d:response>
            <d:href>/</d:href>
            <d:propstat><d:prop>
              <d:current-user-principal><d:href>/123/principal/</d:href></d:current-user-principal>
            </d:prop></d:propstat>
          </d:response>
        </d:multistatus>"""
    )


def _home() -> httpx.Response:
    return _xml(
        """<?xml version="1.0"?>
        <d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
          <d:response>
            <d:href>/123/principal/</d:href>
            <d:propstat><d:prop>
              <c:calendar-home-set><d:href>/123/calendars/</d:href></c:calendar-home-set>
            </d:prop></d:propstat>
          </d:response>
        </d:multistatus>"""
    )


def _calendars() -> httpx.Response:
    return _xml(
        """<?xml version="1.0"?>
        <d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
          <d:response>
            <d:href>/123/calendars/</d:href>
            <d:propstat><d:prop>
              <d:displayname>calendars</d:displayname>
              <d:resourcetype><d:collection/></d:resourcetype>
            </d:prop></d:propstat>
          </d:response>
          <d:response>
            <d:href>/123/calendars/personal/</d:href>
            <d:propstat><d:prop>
              <d:displayname>Personal</d:displayname>
              <d:resourcetype><d:collection/><c:calendar/></d:resourcetype>
            </d:prop></d:propstat>
          </d:response>
          <d:response>
            <d:href>/123/calendars/home/</d:href>
            <d:propstat><d:prop>
              <d:displayname>Home</d:displayname>
              <d:resourcetype><d:collection/><c:calendar/></d:resourcetype>
            </d:prop></d:propstat>
          </d:response>
        </d:multistatus>"""
    )


def _report(ics: str = _ICS) -> httpx.Response:
    return _xml(
        f"""<?xml version="1.0"?>
        <d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
          <d:response>
            <d:href>/123/calendars/home/event.ics</d:href>
            <d:propstat><d:prop><c:calendar-data>{ics}</c:calendar-data></d:prop></d:propstat>
          </d:response>
        </d:multistatus>"""
    )


def test_duration_is_the_end_when_dtend_is_absent() -> None:
    found = events_on(
        """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:lunch
DTSTART:20260922T160000Z
DURATION:PT90M
SUMMARY:Lunch
END:VEVENT
END:VCALENDAR
""",
        day=date(2026, 9, 22),
        zone=ZoneInfo("UTC"),
    )

    assert len(found) == 1
    assert found[0].summary == "Lunch"
    assert found[0].all_day is False
    assert found[0].end == datetime(2026, 9, 22, 17, 30, tzinfo=UTC)


def test_event_with_both_end_and_duration_is_skipped() -> None:
    found = events_on(
        """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:bad
DTSTART:20260922T160000Z
DTEND:20260922T170000Z
DURATION:PT90M
SUMMARY:Broken
END:VEVENT
BEGIN:VEVENT
UID:ok
DTSTART:20260922T180000Z
DTEND:20260922T190000Z
SUMMARY:Kept
END:VEVENT
END:VCALENDAR
""",
        day=date(2026, 9, 22),
        zone=ZoneInfo("UTC"),
    )

    assert [event.summary for event in found] == ["Kept"]


def test_cancelled_events_drop_out_and_all_day_stays() -> None:
    found = events_on(_ICS, day=datetime(2026, 9, 22).date(), zone=ZoneInfo("America/Chicago"))
    assert [event.summary for event in found] == ["Trip", "Dentist"]
    assert found[0].all_day is True
    assert found[1].location == "Office"


def test_icloud_discovery_hosts_are_allowed() -> None:
    assert host_allowed("p12-caldav.icloud.com", "caldav.icloud.com")
    assert host_allowed("caldav.test", "caldav.test")
    assert not host_allowed("evil.example", "caldav.test")
    assert not host_allowed("p12-caldav.icloud.com", "caldav.test")


async def test_agenda_reads_the_named_calendar_for_the_local_day() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert _PASSWORD not in str(request.url)
        assert _PASSWORD.encode() not in request.content
        path = request.url.path
        if request.method == "PROPFIND" and path == "/":
            return _principal()
        if request.method == "PROPFIND" and path.endswith("/principal/"):
            return _home()
        if request.method == "PROPFIND" and path.endswith("/calendars/"):
            return _calendars()
        if request.method == "REPORT":
            assert path.endswith("/calendars/home/")
            assert b"expand" in request.content
            assert b"20260922T050000Z" in request.content
            assert b"20260923T050000Z" in request.content
            return _report()
        raise AssertionError(f"{request.method} {path}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    client = CalendarClient(_configured(), http=http)
    agenda = await client.agenda(datetime(2026, 9, 22).date())
    await http.aclose()

    assert agenda.day.isoformat() == "2026-09-22"
    assert [event.summary for event in agenda.events] == ["Trip", "Dentist"]
    assert agenda.events[1].start.isoformat() == "2026-09-22T09:00:00-05:00"
    assert not any("personal" in request.url.path for request in seen)
    assert seen[0].headers["authorization"].startswith("Basic")


async def test_omitted_date_uses_the_household_timezone() -> None:
    """04:00 UTC is still the previous evening in Chicago."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "REPORT":
            assert b"20260921T050000Z" in request.content
            assert b"20260922T050000Z" in request.content
            return _report("BEGIN:VCALENDAR\nEND:VCALENDAR\n")
        if request.url.path == "/":
            return _principal()
        if request.url.path.endswith("/principal/"):
            return _home()
        return _calendars()

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    client = CalendarClient(_configured(), http=http)
    agenda = await client.agenda(now=datetime(2026, 9, 22, 4, 0, tzinfo=UTC))
    await http.aclose()
    assert agenda.day.isoformat() == "2026-09-21"


async def test_a_redirect_off_the_calendar_host_is_not_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "caldav.test"
        return httpx.Response(302, headers={"location": "http://127.0.0.1/latest"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    client = CalendarClient(_configured(), http=http)
    with pytest.raises(CalendarError, match="not allowed"):
        await client.agenda(datetime(2026, 9, 22).date())
    await http.aclose()


async def test_login_failure_does_not_include_the_password() -> None:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(401)),
        trust_env=False,
    )
    client = CalendarClient(_configured(), http=http)
    with pytest.raises(CalendarError, match="login was rejected") as caught:
        await client.agenda(datetime(2026, 9, 22).date())
    await http.aclose()
    assert _PASSWORD not in str(caught.value)


async def test_missing_calendar_config_is_an_error() -> None:
    client = CalendarClient(_configured(caldav_password=""))
    with pytest.raises(CalendarError, match="not configured"):
        await client.agenda(datetime(2026, 9, 22).date())


async def test_two_calendars_with_the_same_name_are_not_guessed() -> None:
    listed = _calendars().text.replace(
        "<d:displayname>Personal</d:displayname>",
        "<d:displayname>Home</d:displayname>",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "REPORT":
            raise AssertionError("must not query until the name is unique")
        if request.url.path == "/":
            return _principal()
        if request.url.path.endswith("/principal/"):
            return _home()
        return httpx.Response(207, content=listed.encode())

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    client = CalendarClient(_configured(), http=http)
    with pytest.raises(CalendarError, match="more than one"):
        await client.agenda(datetime(2026, 9, 22).date())
    await http.aclose()


async def test_tool_rejects_a_bad_date_and_returns_the_agenda() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "REPORT":
            return _report()
        if request.url.path == "/":
            return _principal()
        if request.url.path.endswith("/principal/"):
            return _home()
        return _calendars()

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    handlers = bind_calendar(_configured(), client=CalendarClient(_configured(), http=http))
    with pytest.raises(CalendarError, match="YYYY-MM-DD"):
        await handlers[CALENDAR_TODAY]({"date": "tomorrow"})
    raw = await handlers[CALENDAR_TODAY]({"date": "2026-09-22"})
    await http.aclose()
    payload = json.loads(raw)
    assert payload["calendar"] == "Home"
    assert payload["date"] == "2026-09-22"
    assert [event["summary"] for event in payload["events"]] == ["Trip", "Dentist"]
    assert payload["events"][0]["all_day"] is True
    assert payload["events"][0]["start"] == "2026-09-22"
    assert payload["events"][0]["end"] == "2026-09-22"


def test_builtin_catalog_marks_calendar_untrusted_and_not_web() -> None:
    meta = builtin_catalog().meta(CALENDAR_TODAY)
    assert meta.trusted is False
    assert meta.web is False
    assert meta.staging is False
    assert meta.sink is None


async def test_calendar_does_not_require_web_access_and_taints() -> None:
    jacob = agent("jacob", "+15555550101", tools=("calendar_today",), web_access=False)
    catalog = builtin_catalog()
    catalog.bind(CALENDAR_TODAY, lambda _a: json.dumps({"events": []}))
    surface = ToolBroker(
        catalog, household(jacob, agent("spouse", "+15555550102")).broker
    ).for_agent(jacob)
    assert [spec.name for spec in surface.specs()] == ["calendar_today"]
    await surface.execute(tool_call(name="calendar_today", arguments="{}"))
    assert surface.tainted is True


async def test_report_only_keeps_the_calendar_read() -> None:
    jacob = agent(
        "jacob",
        "+15555550101",
        tools=("get_time", "calendar_today", "relay", "job_create"),
    )
    surface = ToolBroker(
        builtin_catalog(), household(jacob, agent("spouse", "+15555550102")).broker
    ).for_agent(jacob, report_only=True)
    assert {spec.name for spec in surface.specs()} == {"get_time", "calendar_today"}


async def test_a_long_day_is_truncated() -> None:
    events = "\n".join(
        f"""BEGIN:VEVENT
UID:{i}
DTSTART:20260922T120000Z
DTEND:20260922T121500Z
SUMMARY:Event {i}
END:VEVENT"""
        for i in range(45)
    )
    ics = f"BEGIN:VCALENDAR\n{events}\nEND:VCALENDAR\n"
    found = events_on(ics, day=datetime(2026, 9, 22).date(), zone=ZoneInfo("America/Chicago"))
    assert len(found) == 45

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "REPORT":
            return _report(ics)
        if request.url.path == "/":
            return _principal()
        if request.url.path.endswith("/principal/"):
            return _home()
        return _calendars()

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    agenda = await CalendarClient(_configured(), http=http).agenda(datetime(2026, 9, 22).date())
    await http.aclose()
    assert agenda.truncated is True
    assert len(agenda.events) == 40
