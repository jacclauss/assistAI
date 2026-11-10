from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tests.clock_fakes import CHICAGO, chicago

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
