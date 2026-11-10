from __future__ import annotations

import pytest

from assistai.sysclock import ClockState, kernel_clock_state, state_from_adjtimex


@pytest.mark.parametrize(
    ("result", "state"),
    [
        (0, ClockState.SYNCED),
        (1, ClockState.SYNCED),
        (4, ClockState.SYNCED),
        (5, ClockState.UNSYNCED),
        (-1, ClockState.UNKNOWN),
        (6, ClockState.UNKNOWN),
    ],
)
def test_adjtimex_results_map_to_a_state(result: int, state: ClockState) -> None:
    """TIME_ERROR is the only unsynced answer; a refused call is not a no."""
    assert state_from_adjtimex(result) is state


def test_a_kernel_without_adjtimex_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("assistai.sysclock.sys.platform", "darwin")

    assert kernel_clock_state() is ClockState.UNKNOWN


def test_the_real_kernel_query_answers_without_raising() -> None:
    """On Linux this reads the host's clock state; elsewhere it is unknown. Either way, no crash."""
    assert kernel_clock_state() in set(ClockState)
