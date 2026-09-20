from __future__ import annotations

from assistai.signal.ratelimit import RateLimiter


def test_allows_up_to_the_limit() -> None:
    ticks = [0.0]
    limiter = RateLimiter(3, 60.0, clock=lambda: ticks[0])

    assert [limiter.allow("+1") for _ in range(4)] == [True, True, True, False]


def test_window_slides() -> None:
    ticks = [0.0]
    limiter = RateLimiter(2, 60.0, clock=lambda: ticks[0])

    assert limiter.allow("+1") is True
    assert limiter.allow("+1") is True
    assert limiter.allow("+1") is False

    ticks[0] = 61
    assert limiter.allow("+1") is True


def test_senders_are_independent() -> None:
    ticks = [0.0]
    limiter = RateLimiter(1, 60.0, clock=lambda: ticks[0])

    assert limiter.allow("+1") is True
    assert limiter.allow("+1") is False
    assert limiter.allow("+2") is True


def test_would_allow_does_not_spend_the_budget() -> None:
    ticks = [0.0]
    limiter = RateLimiter(1, 60.0, clock=lambda: ticks[0])

    assert limiter.would_allow("+1") is True
    assert limiter.would_allow("+1") is True
    assert limiter.allow("+1") is True
    assert limiter.would_allow("+1") is False


def test_record_spends_the_budget_after_success() -> None:
    ticks = [0.0]
    limiter = RateLimiter(1, 60.0, clock=lambda: ticks[0])

    assert limiter.would_allow("+1") is True
    limiter.record("+1")
    assert limiter.would_allow("+1") is False
    assert limiter.allow("+1") is False


def test_key_count_is_bounded() -> None:
    ticks = [0.0]
    limiter = RateLimiter(5, 60.0, max_keys=8, clock=lambda: ticks[0])

    for i in range(200):
        ticks[0] = float(i)
        limiter.allow(f"+{i}")

    assert len(limiter._hits) <= 8
