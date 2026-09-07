"""E.164 phone numbers. Signal identifies peers by these, not by names."""

from __future__ import annotations

import re

_E164 = re.compile(r"\+[1-9]\d{6,14}$")


class InvalidNumberError(ValueError):
    """The value is not a plausible E.164 number."""


def normalize_e164(value: str) -> str:
    """Strip common separators and require a leading ``+`` and country code."""
    compact = re.sub(r"[\s\-().]", "", value.strip())
    if not _E164.fullmatch(compact):
        raise InvalidNumberError(f"not an E.164 number: {value!r}")
    return compact
