"""Model-facing calendar read. One shared calendar, one day, no writes."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta
from typing import Any

from assistai.calendar.client import Agenda, CalendarClient
from assistai.calendar.parse import CalendarEvent
from assistai.config import Settings
from assistai.errors import CalendarError
from assistai.inference.types import ToolSpec

CALENDAR_TODAY = "calendar_today"

CALENDAR_TODAY_SPEC = ToolSpec(
    name=CALENDAR_TODAY,
    description=(
        "Read the shared household calendar for one day. Use this for "
        "'what's on the docket today' and for other single days. "
        "Omit date for today in the household timezone; otherwise pass YYYY-MM-DD. "
        "List only the events returned. An all-day event's end is its last day, "
        "not the following morning. Do not invent events. This does not create "
        "or change events."
    ),
    parameters={
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Local calendar day, YYYY-MM-DD. Omit for today.",
            }
        },
        "additionalProperties": False,
    },
)


def bind_calendar(
    settings: Settings,
    *,
    client: CalendarClient | None = None,
) -> dict[str, Callable[[dict[str, Any]], Awaitable[str]]]:
    """Handler closed over the household calendar. Bound when the channel starts."""
    calendar = client or CalendarClient(settings)

    async def today(arguments: dict[str, Any]) -> str:
        day = _day(arguments.get("date"))
        agenda = await calendar.agenda(day)
        return json.dumps(_payload(agenda), ensure_ascii=False)

    return {CALENDAR_TODAY: today}


def _day(value: object) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CalendarError("date must be YYYY-MM-DD")
    cleaned = value.strip()
    if not cleaned:
        return None
    # fromisoformat accepts more than one layout. The model is told YYYY-MM-DD.
    if len(cleaned) != 10:
        raise CalendarError("date must be YYYY-MM-DD")
    try:
        return date.fromisoformat(cleaned)
    except ValueError as exc:
        raise CalendarError("date must be YYYY-MM-DD") from exc


def _payload(agenda: Agenda) -> dict[str, Any]:
    body: dict[str, Any] = {
        "calendar": agenda.calendar,
        "timezone": agenda.timezone,
        "date": agenda.day.isoformat(),
        "events": [_event_dict(event) for event in agenda.events],
    }
    if agenda.truncated:
        body["truncated"] = True
    return body


def _event_dict(event: CalendarEvent) -> dict[str, Any]:
    # iCalendar all-day DTEND is exclusive. A one-day event stored as the 22nd
    # through the 23rd is the 22nd, and that is the day a person means.
    end = event.end
    if event.all_day:
        end = event.end - timedelta(days=1)
        if end < event.start:
            end = event.start
    payload: dict[str, Any] = {
        "summary": event.summary,
        "all_day": event.all_day,
        "start": _when(event.start, all_day=event.all_day),
        "end": _when(end, all_day=event.all_day),
    }
    if event.location:
        payload["location"] = event.location
    if event.description:
        payload["description"] = event.description
    return payload


def _when(moment: datetime, *, all_day: bool) -> str:
    if all_day:
        return moment.date().isoformat()
    return moment.isoformat()
