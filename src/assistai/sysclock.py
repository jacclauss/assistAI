"""Whether the kernel believes its own clock.

The Pi can boot with a clock that is hours or weeks behind until NTP lands.
Times written then are wrong for good, and a later step forward makes fresh
history look old enough to delete. The gateway asks here before it starts
Signal. This is system state, not a household local-time rule, so it lives
apart from ``assistai.clock``.

The query reads clock state and changes nothing. A kernel or seccomp profile
that refuses it is an answer too: ``UNKNOWN``, and the caller falls back to
what the store remembers.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

# struct timex is 208 bytes on 64-bit glibc. A larger zeroed buffer leaves
# room for any layout, and modes = 0 (its first field) asks for a read only.
_TIMEX_BYTES = 512
_TIME_ERROR = 5


class ClockState(Enum):
    SYNCED = "synced"
    UNSYNCED = "unsynced"
    UNKNOWN = "unknown"


class ClockSync(Protocol):
    def state(self) -> ClockState: ...


@dataclass(frozen=True)
class KernelClockSync:
    """The kernel's view via ``adjtimex``. Linux only; elsewhere ``UNKNOWN``."""

    def state(self) -> ClockState:
        return kernel_clock_state()


def kernel_clock_state() -> ClockState:
    """Read the kernel's clock status without changing it."""
    if sys.platform != "linux":
        return ClockState.UNKNOWN
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        adjtimex = libc.adjtimex
        adjtimex.argtypes = [ctypes.c_void_p]
        adjtimex.restype = ctypes.c_int
        result = adjtimex(ctypes.create_string_buffer(_TIMEX_BYTES))
    except (OSError, AttributeError):
        return ClockState.UNKNOWN
    return state_from_adjtimex(int(result))


def state_from_adjtimex(result: int) -> ClockState:
    """Map an ``adjtimex`` return value to a state.

    ``TIME_ERROR`` means the clock is not synchronized. ``-1`` means the call
    was refused. The leap-second states 0 to 4 all mean synchronized.
    """
    if result == _TIME_ERROR:
        return ClockState.UNSYNCED
    if 0 <= result < _TIME_ERROR:
        return ClockState.SYNCED
    return ClockState.UNKNOWN
