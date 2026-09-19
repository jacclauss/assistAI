from __future__ import annotations

import pytest

from assistai.signal.numbers import InvalidNumberError, is_uuid, normalize_e164


def test_strips_separators() -> None:
    assert normalize_e164("+1 (555) 555-0101") == "+15555550101"


def test_rejects_missing_plus() -> None:
    with pytest.raises(InvalidNumberError):
        normalize_e164("15555550101")


def test_rejects_too_short() -> None:
    with pytest.raises(InvalidNumberError):
        normalize_e164("+123")


def test_uuid_shape() -> None:
    assert is_uuid("429cce0e-9174-4d7a-a98b-1cb9208b1951")
    assert is_uuid("429CCE0E-9174-4D7A-A98B-1CB9208B1951")
    assert not is_uuid("not-a-uuid")
    assert not is_uuid("+15555550101")
