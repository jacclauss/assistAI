from __future__ import annotations

import random
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from assistai.clock import (
    LocalAt,
    Schedule,
    fix_at,
    local_reading,
    next_slot,
    resolve_at,
    say_day,
    say_duration,
    say_repeat,
    say_when,
)
from assistai.errors import JobError
from tests.clock_fakes import CHICAGO, FakeClock, chicago, chicago_clock, wall

CDT = timedelta(hours=-5)
CST = timedelta(hours=-6)


def test_the_zone_database_knows_the_2026_and_2027_transitions() -> None:
    """Every DST test below trusts these offsets. Missing or stale zone data fails here first."""
    first = datetime(2026, 11, 1, 1, 30, tzinfo=CHICAGO)
    second = datetime(2026, 11, 1, 1, 30, tzinfo=CHICAGO, fold=1)
    assert first.utcoffset() == CDT
    assert second.utcoffset() == CST
    assert chicago(2026, 11, 1, 1, 30, fold=1) - chicago(2026, 11, 1, 1, 30) == 3600

    assert datetime(2027, 3, 14, 1, 59, tzinfo=CHICAGO).utcoffset() == CST
    assert datetime(2027, 3, 14, 3, 30, tzinfo=CHICAGO).utcoffset() == CDT

    # 2:30 AM on spring-forward day never shows on a clock: it comes back as another time.
    gap = datetime(2027, 3, 14, 2, 30, tzinfo=CHICAGO)
    back = datetime.fromtimestamp(gap.timestamp(), UTC).astimezone(CHICAGO)
    assert (back.hour, back.minute) != (2, 30)


DAY = 86400


def _daily(anchor: float, at_local: str, every_days: int = 1) -> Schedule:
    return Schedule(zone=CHICAGO, anchor_at=anchor, every_days=every_days, at_local=at_local)


def test_daily_eight_am_stays_eight_am_across_fall_back() -> None:
    """A slow run finishing late does not move the next 8:00, and Nov 1 is 25 hours on."""
    schedule = _daily(chicago(2026, 10, 30, 8), "08:00")

    saturday = next_slot(schedule, after=schedule.anchor_at + 7)
    sunday = next_slot(schedule, after=saturday + 300)
    monday = next_slot(schedule, after=sunday + 3 * 3600)

    assert wall(saturday) == "2026-10-31 08:00:00 CDT"
    assert wall(sunday) == "2026-11-01 08:00:00 CST"
    assert wall(monday) == "2026-11-02 08:00:00 CST"
    assert sunday - saturday == 25 * 3600
    assert monday - sunday == 24 * 3600


def test_daily_eight_am_stays_eight_am_across_spring_forward() -> None:
    schedule = _daily(chicago(2027, 3, 13, 8), "08:00")

    sunday = next_slot(schedule, after=schedule.anchor_at)
    monday = next_slot(schedule, after=sunday + 60)

    assert wall(sunday) == "2027-03-14 08:00:00 CDT"
    assert wall(monday) == "2027-03-15 08:00:00 CDT"
    assert sunday - schedule.anchor_at == 23 * 3600


def test_at_in_the_spring_gap_is_refused_with_the_reason() -> None:
    clock = chicago_clock(chicago(2027, 3, 13, 12))

    for at in (
        LocalAt(date=date(2027, 3, 14), hour=2, minute=30),
        LocalAt(day="tomorrow", hour=2, minute=30),
        LocalAt(day="sunday", hour=2, minute=30),
    ):
        with pytest.raises(JobError) as caught:
            resolve_at(at, clock=clock)
        assert "2:30 AM does not exist on Sunday, March 14, 2027" in str(caught.value)
        assert "2:00 to 3:00" in str(caught.value)


def test_a_bare_gap_time_already_behind_today_is_tomorrow() -> None:
    """At 5 AM on spring-forward day the skipped 2:30 is gone, so "at 2:30" is Monday's."""
    clock = chicago_clock(chicago(2027, 3, 14, 5))

    assert wall(resolve_at(LocalAt(hour=2, minute=30), clock=clock)) == "2027-03-15 02:30:00 CDT"


def test_a_slot_in_the_spring_gap_runs_an_hour_later_that_day_only() -> None:
    schedule = _daily(chicago(2027, 3, 13, 2, 30), "02:30")

    sunday = next_slot(schedule, after=schedule.anchor_at)
    monday = next_slot(schedule, after=sunday)

    assert wall(schedule.anchor_at) == "2027-03-13 02:30:00 CST"
    assert wall(sunday) == "2027-03-14 03:30:00 CDT"
    assert wall(monday) == "2027-03-15 02:30:00 CDT"


def test_a_slot_in_the_fall_back_hour_runs_once() -> None:
    schedule = _daily(chicago(2026, 10, 31, 1, 30), "01:30")

    sunday = next_slot(schedule, after=schedule.anchor_at)
    after_first = next_slot(schedule, after=sunday)
    during_second = next_slot(schedule, after=chicago(2026, 11, 1, 1, 15, fold=1))

    assert wall(sunday) == "2026-11-01 01:30:00 CDT"
    assert wall(after_first) == "2026-11-02 01:30:00 CST"
    assert during_second == after_first


def test_one_forty_at_one_twenty_cst_on_fall_back_day_is_twenty_minutes_away() -> None:
    now = chicago(2026, 11, 1, 1, 20, fold=1)
    clock = chicago_clock(now)

    for at in (LocalAt(hour=1, minute=40), LocalAt(day="today", hour=1, minute=40)):
        moment = resolve_at(at, clock=clock)
        assert wall(moment) == "2026-11-01 01:40:00 CST"
        assert moment - now == 20 * 60


def test_a_repeated_time_is_the_first_unless_it_passed_or_fold_says_second() -> None:
    clock = chicago_clock(chicago(2026, 11, 1, 1, 20))

    first = resolve_at(LocalAt(hour=1, minute=40), clock=clock)
    second = resolve_at(LocalAt(hour=1, minute=40, fold=1), clock=clock)

    assert wall(first) == "2026-11-01 01:40:00 CDT"
    assert wall(second) == "2026-11-01 01:40:00 CST"
    with pytest.raises(JobError, match="already passed"):
        resolve_at(
            LocalAt(day="today", hour=1, minute=10, fold=0),
            clock=chicago_clock(chicago(2026, 11, 1, 1, 20, fold=1)),
        )


def test_tomorrow_at_eleven_thirty_pm_is_the_local_tomorrow() -> None:
    now = chicago(2026, 10, 30, 23, 30)
    assert datetime.fromtimestamp(now, UTC).date() == date(2026, 10, 31)
    clock = chicago_clock(now)

    moment = resolve_at(LocalAt(day="tomorrow", hour=8), clock=clock)

    assert wall(moment) == "2026-10-31 08:00:00 CDT"
    assert say_when(moment, clock=clock) == "tomorrow at 8:00 AM (Saturday, October 31)"


def test_monday_at_seven_thirty_on_monday_morning_is_today() -> None:
    clock = chicago_clock(chicago(2026, 11, 2, 6))

    moment = resolve_at(LocalAt(day="monday", hour=7, minute=30), clock=clock)

    assert wall(moment) == "2026-11-02 07:30:00 CST"
    assert say_when(moment, clock=clock) == "today at 7:30 AM"


def test_a_weekday_already_passed_today_is_next_week() -> None:
    monday = LocalAt(day="monday", hour=7, minute=30)

    later = resolve_at(monday, clock=chicago_clock(chicago(2026, 11, 2, 9)))
    too_close = resolve_at(monday, clock=chicago_clock(chicago(2026, 11, 2, 7, 30) - 30))
    from_friday = resolve_at(monday, clock=chicago_clock(chicago(2026, 10, 30, 12)))

    assert wall(later) == "2026-11-09 07:30:00 CST"
    assert wall(too_close) == "2026-11-09 07:30:00 CST"
    assert wall(from_friday) == "2026-11-02 07:30:00 CST"


def test_a_time_more_than_a_minute_past_is_refused_with_the_local_now() -> None:
    clock = chicago_clock(chicago(2026, 10, 30, 23, 30))

    for at in (
        LocalAt(date=date(2026, 10, 30), hour=8),
        LocalAt(day="today", hour=23, minute=28),
    ):
        with pytest.raises(JobError) as caught:
            resolve_at(at, clock=clock)
        message = str(caught.value)
        assert "already passed" in message
        assert "It is now Friday, October 30, 2026 at 11:30 PM" in message
        assert "America/Chicago" in message


def test_a_time_within_a_minute_past_is_kept() -> None:
    eight = chicago(2026, 10, 30, 8)
    clock = chicago_clock(eight + 45)

    assert resolve_at(LocalAt(day="today", hour=8), clock=clock) == eight
    assert resolve_at(LocalAt(date=date(2026, 10, 30), hour=8), clock=clock) == eight
    assert resolve_at(LocalAt(hour=8), clock=clock) == eight


def test_a_bare_time_already_passed_is_tomorrow() -> None:
    clock = chicago_clock(chicago(2026, 10, 30, 8, 2))

    assert wall(resolve_at(LocalAt(hour=8), clock=clock)) == "2026-10-31 08:00:00 CDT"
    assert wall(resolve_at(LocalAt(hour=20), clock=clock)) == "2026-10-30 20:00:00 CDT"


@pytest.mark.parametrize(
    "at",
    [
        LocalAt(hour=24),
        LocalAt(hour=8, minute=60),
        LocalAt(hour=8, fold=2),
        LocalAt(day="someday", hour=8),
        LocalAt(day="today", date=date(2026, 10, 30), hour=8),
    ],
)
def test_a_malformed_time_is_refused(at: LocalAt) -> None:
    with pytest.raises(JobError):
        resolve_at(at, clock=chicago_clock(chicago(2026, 10, 30, 7)))


def test_next_slot_every_seconds_stays_on_its_grid_across_fall_back() -> None:
    anchor = chicago(2026, 10, 31, 23)
    schedule = Schedule(zone=CHICAGO, anchor_at=anchor, every_seconds=5400)

    assert next_slot(schedule, after=anchor - DAY) == anchor
    assert next_slot(schedule, after=anchor) == anchor + 5400
    across = next_slot(schedule, after=chicago(2026, 11, 1, 1, 50, fold=1))
    assert (across - anchor) % 5400 == 0
    assert wall(across) == "2026-11-01 02:30:00 CST"
    assert next_slot(schedule, after=across - 1) == across


def test_next_slot_weekly_keeps_its_weekday_and_hour() -> None:
    schedule = _daily(chicago(2026, 12, 10, 8), "08:00", every_days=7)

    assert wall(next_slot(schedule, after=schedule.anchor_at)) == "2026-12-17 08:00:00 CST"
    assert wall(next_slot(schedule, after=chicago(2027, 3, 1))) == "2027-03-04 08:00:00 CST"
    assert wall(next_slot(schedule, after=chicago(2027, 3, 12))) == "2027-03-18 08:00:00 CDT"
    assert next_slot(schedule, after=schedule.anchor_at - 1) == schedule.anchor_at


def test_a_schedule_must_be_well_formed() -> None:
    anchor = chicago(2026, 10, 30, 8)
    with pytest.raises(JobError):
        _daily(anchor, "08:00", every_days=0)
    with pytest.raises(JobError):
        _daily(anchor, "8:00")
    with pytest.raises(JobError):
        _daily(anchor, "24:00")
    with pytest.raises(JobError):
        Schedule(zone=CHICAGO, anchor_at=anchor, every_seconds=0)
    with pytest.raises(JobError):
        Schedule(zone=CHICAGO, anchor_at=anchor, every_days=1, every_seconds=60, at_local="08:00")
    with pytest.raises(JobError):
        Schedule(zone=CHICAGO, anchor_at=anchor, every_days=1)
    with pytest.raises(JobError, match="no next run"):
        next_slot(Schedule(zone=CHICAGO, anchor_at=anchor), after=anchor)


def _transition_days(year: int) -> tuple[date, date]:
    """US rule: second Sunday of March, first Sunday of November."""
    march = date(year, 3, 1)
    november = date(year, 11, 1)
    spring = march + timedelta(days=(6 - march.weekday()) % 7 + 7)
    fall = november + timedelta(days=(6 - november.weekday()) % 7)
    return spring, fall


def _wall_slot(day: date, hour: int, minute: int) -> float:
    return datetime.combine(day, time(hour, minute), tzinfo=CHICAGO).timestamp()


def test_next_slot_holds_its_rules_for_many_schedules() -> None:
    rng = random.Random(16_2026)  # noqa: S311 - a fixed seed, not a secret
    transitions = [day for year in range(2026, 2031) for day in _transition_days(year)]
    assert _transition_days(2026)[1] == date(2026, 11, 1)
    assert _transition_days(2027)[0] == date(2027, 3, 14)
    start, span = date(2026, 1, 1), (date(2030, 12, 31) - date(2026, 1, 1)).days
    shifted = repeated = elapsed = 0

    for _ in range(2000):
        if rng.random() < 0.3:
            step = rng.choice([60, 300, 900, 3600, 5400, 7200, DAY, rng.randint(60, 7 * DAY)])
            anchor = float(int(_wall_slot(start + timedelta(days=rng.randint(0, span)), 0, 0)))
            anchor += rng.randint(0, DAY - 1)
            schedule = Schedule(zone=CHICAGO, anchor_at=anchor, every_seconds=step)
            after = anchor + rng.uniform(-2 * DAY, 2 * 365 * DAY)
            slot = next_slot(schedule, after=after)
            assert slot > after
            assert (slot - anchor) % step == 0
            assert slot == anchor or slot - step <= after
            elapsed += 1
            continue

        every = rng.randint(1, 14)
        hour = rng.choice([1, 1, 2, 2, rng.randint(0, 23)])
        minute = rng.randint(0, 59)
        target: date | None = None
        back = 0
        anchor_date: date = start + timedelta(days=rng.randint(0, span))
        if rng.random() < 0.5:
            chosen = rng.choice(transitions)
            back = rng.randint(0, 30)
            anchor_date = chosen - timedelta(days=back * every)
            target = chosen
        anchor = _wall_slot(anchor_date, hour, minute)
        schedule = _daily(anchor, f"{hour:02d}:{minute:02d}", every_days=every)
        if target is None:
            after = anchor + rng.uniform(-3 * DAY, 4 * 365 * DAY)
        elif back == 0:
            after = anchor - rng.uniform(1, 3 * DAY)
        else:
            after = _wall_slot(target - timedelta(days=every), hour, minute) + rng.uniform(0, 3600)

        slot = next_slot(schedule, after=after)
        local = datetime.fromtimestamp(slot, CHICAGO)
        day = local.date()
        label = f"{schedule} after {wall(after)} gave {wall(slot)}"

        assert slot > after, label
        assert (day - anchor_date).days >= 0, label
        assert (day - anchor_date).days % every == 0, label
        if target is not None:
            assert day == target, label
        if day > anchor_date:
            assert _wall_slot(day - timedelta(days=every), hour, minute) <= after, label
        else:
            assert slot == anchor, label

        first = datetime.combine(day, time(hour, minute), tzinfo=CHICAGO)
        second = first.replace(fold=1)
        first_offset, second_offset = first.utcoffset(), second.utcoffset()
        assert first_offset is not None and second_offset is not None
        if first_offset < second_offset:
            assert (local.hour, local.minute) == (hour + 1, minute), label
            shifted += 1
        else:
            assert (local.hour, local.minute) == (hour, minute), label
            if first_offset > second_offset:
                assert local.utcoffset() == first_offset, label
                repeated += 1

        following = next_slot(schedule, after=slot)
        assert (datetime.fromtimestamp(following, CHICAGO).date() - day).days == every, label

    # The loop proves nothing about DST unless it lands on DST days.
    assert shifted > 20
    assert repeated > 20
    assert elapsed > 300


@pytest.mark.parametrize(
    ("moment", "said"),
    [
        (chicago(2026, 10, 30, 20), "today at 8:00 PM"),
        (chicago(2026, 10, 30, 23, 59), "today at 11:59 PM"),
        (chicago(2026, 10, 31, 0), "tomorrow at 12:00 AM (Saturday, October 31)"),
        (chicago(2026, 10, 31, 8), "tomorrow at 8:00 AM (Saturday, October 31)"),
        (chicago(2026, 11, 3, 7, 30), "Tuesday at 7:30 AM (November 3)"),
        (chicago(2026, 11, 5, 12), "Thursday at 12:00 PM (November 5)"),
        (chicago(2026, 11, 6, 8), "Friday, November 6 at 8:00 AM"),
        (chicago(2026, 12, 10, 8), "Thursday, December 10 at 8:00 AM"),
        (chicago(2027, 12, 10, 8), "Friday, December 10, 2027 at 8:00 AM"),
        (chicago(2026, 10, 29, 9), "Thursday, October 29 at 9:00 AM"),
    ],
)
def test_say_when_reads_relative_to_the_household_today(moment: float, said: str) -> None:
    assert say_when(moment, clock=chicago_clock(chicago(2026, 10, 30, 15))) == said


def test_say_when_names_the_zone_for_a_time_that_happens_twice() -> None:
    clock = chicago_clock(chicago(2026, 11, 1, 1, 20, fold=1))

    assert say_when(chicago(2026, 11, 1, 1, 40, fold=1), clock=clock) == "today at 1:40 AM CST"
    assert say_when(chicago(2026, 11, 1, 1, 40), clock=clock) == "today at 1:40 AM CDT"
    assert say_when(chicago(2026, 11, 1, 2, 40), clock=clock) == "today at 2:40 AM"


def test_say_when_follows_a_moving_clock() -> None:
    fake = FakeClock(chicago(2026, 10, 30, 15))
    clock = chicago_clock(fake)
    moment = chicago(2026, 10, 31, 8)

    assert say_when(moment, clock=clock).startswith("tomorrow")
    fake.set(chicago(2026, 10, 31, 1))
    assert say_when(moment, clock=clock) == "today at 8:00 AM"


def test_say_repeat_names_each_shape() -> None:
    clock = chicago_clock(chicago(2026, 10, 30, 15))
    anchor = chicago(2026, 12, 10, 8)

    assert say_repeat(_daily(anchor, "08:00"), clock=clock) == "every day at 8:00 AM"
    assert say_repeat(_daily(anchor, "08:00", 2), clock=clock) == "every 2 days at 8:00 AM"
    assert say_repeat(_daily(anchor, "08:00", 7), clock=clock) == "every Thursday at 8:00 AM"
    assert say_repeat(_daily(anchor, "20:05", 14), clock=clock) == "every 14 days at 8:05 PM"
    for seconds, said in (
        (5400, "every 90 minutes"),
        (7200, "every 2 hours"),
        (3600, "every hour"),
    ):
        schedule = Schedule(zone=CHICAGO, anchor_at=anchor, every_seconds=seconds)
        assert say_repeat(schedule, clock=clock) == said
    assert say_repeat(Schedule(zone=CHICAGO, anchor_at=anchor), clock=clock) == "once"


def test_say_repeat_names_a_zone_that_is_not_the_household_one() -> None:
    clock = chicago_clock(chicago(2026, 10, 30, 15))
    schedule = Schedule(
        zone=ZoneInfo("Europe/London"),
        anchor_at=chicago(2026, 12, 10, 8),
        every_days=1,
        at_local="14:00",
    )

    assert say_repeat(schedule, clock=clock) == "every day at 2:00 PM Europe/London"


@pytest.mark.parametrize(
    ("seconds", "said"),
    [
        (60, "1 minute"),
        (120, "2 minutes"),
        (5400, "90 minutes"),
        (3600, "1 hour"),
        (7200, "2 hours"),
        (DAY, "1 day"),
        (50 * DAY, "50 days"),
        (90, "90 seconds"),
    ],
)
def test_say_duration_uses_the_largest_whole_unit(seconds: int, said: str) -> None:
    """The relay preview's delay wording, which moved here from relay."""
    assert say_duration(seconds) == said


@pytest.mark.parametrize(
    ("day", "said"),
    [
        (date(2026, 10, 30), "today, Friday, October 30"),
        (date(2026, 10, 29), "yesterday, Thursday, October 29"),
        (date(2026, 10, 31), "tomorrow, Saturday, October 31"),
        (date(2026, 11, 2), "Monday, November 2"),
        (date(2027, 1, 4), "Monday, January 4, 2027"),
    ],
)
def test_say_day_names_the_day_against_today(day: date, said: str) -> None:
    assert say_day(day, clock=chicago_clock(chicago(2026, 10, 30, 15))) == said


def test_the_household_today_is_not_the_utc_date_in_the_evening() -> None:
    clock = chicago_clock(chicago(2026, 10, 30, 23, 30))

    assert clock.today() == date(2026, 10, 30)
    assert say_day(date(2026, 10, 31), clock=clock) == "tomorrow, Saturday, October 31"


def test_a_local_reading_names_the_offset_on_each_side_of_fall_back() -> None:
    """The operator log must show which 1:30 the gateway believes it is."""
    clock = chicago_clock(0.0)

    assert local_reading(chicago(2026, 11, 1, 1, 30), clock=clock) == (
        "2026-11-01T01:30:00-05:00",
        "-05:00",
    )
    assert local_reading(chicago(2026, 11, 1, 1, 30, fold=1), clock=clock) == (
        "2026-11-01T01:30:00-06:00",
        "-06:00",
    )


@pytest.mark.parametrize(
    ("now", "at", "fixed"),
    [
        pytest.param(
            chicago(2026, 10, 30, 23, 58),
            LocalAt(day="tomorrow", hour=8),
            LocalAt(date=date(2026, 10, 31), hour=8),
            id="tomorrow before midnight",
        ),
        pytest.param(
            chicago(2026, 10, 30, 21, 12),
            LocalAt(hour=8),
            LocalAt(date=date(2026, 10, 31), hour=8),
            id="a bare hour already past today",
        ),
        pytest.param(
            chicago(2026, 10, 31, 22),
            LocalAt(hour=1, minute=30),
            LocalAt(date=date(2026, 11, 1), hour=1, minute=30, fold=0),
            id="the first of a repeated hour",
        ),
        pytest.param(
            chicago(2026, 11, 1, 1, 20, fold=1),
            LocalAt(hour=1, minute=40),
            LocalAt(date=date(2026, 11, 1), hour=1, minute=40, fold=1),
            id="the second of a repeated hour",
        ),
    ],
)
def test_fix_at_pins_the_date_and_the_occurrence(now: float, at: LocalAt, fixed: LocalAt) -> None:
    """What is staged resolves to the same instant at the yes, whatever day the yes is."""
    clock = chicago_clock(now)

    assert fix_at(at, clock=clock) == fixed
    assert resolve_at(fixed, clock=clock) == resolve_at(at, clock=clock)
