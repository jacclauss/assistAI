from __future__ import annotations

import pytest

from assistai.signal.numbers import InvalidNumberError, normalize_e164


def test_strips_separators() -> None:
    assert normalize_e164("+1 (555) 555-0101") == "+15555550101"


def test_rejects_missing_plus() -> None:
    with pytest.raises(InvalidNumberError):
        normalize_e164("15555550101")


def test_rejects_too_short() -> None:
    with pytest.raises(InvalidNumberError):
        normalize_e164("+123")
