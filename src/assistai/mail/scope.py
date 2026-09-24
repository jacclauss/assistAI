"""The only Gmail scope this process is allowed to hold.

``gmail.modify`` is the narrowest scope that can change labels. Google's
grant also authorizes send. This process never calls send or permanent
delete, and the client refuses those URL paths.
"""

from __future__ import annotations

GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.modify"

_FORBIDDEN = ("gmail.send", "gmail.compose", "gmail.metadata", "/send", "/delete")


def assert_scope_allowed(scope: str) -> None:
    if scope != GMAIL_SCOPE or any(piece in scope for piece in _FORBIDDEN):
        raise RuntimeError("gmail scope is not the modify-only grant")
