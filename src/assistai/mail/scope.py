"""The only Gmail scope this process is allowed to hold.

``gmail.modify`` reads, labels, archives, stars, moves to Trash, and creates
drafts. It does not include ``gmail.send`` or permanent delete. Those scopes
are never requested, and the client refuses those URL paths.
"""

from __future__ import annotations

GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.modify"

_FORBIDDEN = ("gmail.send", "gmail.compose", "gmail.metadata", "/send", "/delete")


def assert_scope_allowed(scope: str) -> None:
    if scope != GMAIL_SCOPE or any(piece in scope for piece in _FORBIDDEN):
        raise RuntimeError("gmail scope is not the modify-only grant")
