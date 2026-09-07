"""E.164 phone numbers. Signal identifies peers by these, not by names."""

from __future__ import annotations

import re

from assistai.errors import AssistAIError

_E164 = re.compile(r"\+[1-9]\d{6,14}$")


class InvalidNumberError(AssistAIError, ValueError):
    """The value is not a plausible E.164 number.

    Also an ``AssistAIError`` so a typo at the CLI prints one line and exits 2
    instead of a traceback. Still a ``ValueError`` for callers that catch that.
    """


def normalize_e164(value: str) -> str:
    """Strip common separators and require a leading ``+`` and country code."""
    compact = re.sub(r"[\s\-().]", "", value.strip())
    if not _E164.fullmatch(compact):
        raise InvalidNumberError(f"not an E.164 number: {value!r}")
    return compact
