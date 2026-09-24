"""Gmail API over httpx. One person's token, no send, no permanent delete."""

from __future__ import annotations

import base64
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlencode

import httpx

from assistai.config import Settings
from assistai.errors import MailError
from assistai.mail.actions import DraftMessage, FileBatch
from assistai.mail.scope import GMAIL_SCOPE, assert_scope_allowed

GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
MAX_LIST = 15
MAX_BODY = 4000
_HEADER = {"From", "Subject", "Date"}


class TokenStore(Protocol):
    def gmail_token(self, agent: str) -> tuple[str, str, float] | None: ...

    def save_gmail_token(
        self,
        agent: str,
        *,
        refresh_token: str,
        access_token: str,
        expires_at: float,
    ) -> None: ...


@dataclass(frozen=True)
class ListedMail:
    id: str
    sender: str
    subject: str
    date: str
    snippet: str


class GmailClient:
    """Calls Gmail as one household member. A missing token is an error, not a fallback."""

    def __init__(
        self,
        settings: Settings,
        store: TokenStore,
        *,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        assert_scope_allowed(GMAIL_SCOPE)
        self._settings = settings
        self._store = store
        self._http = http or httpx.AsyncClient(timeout=20.0, trust_env=False)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def inbox(self, agent: str, *, query: str = "") -> list[ListedMail]:
        params = {"maxResults": str(MAX_LIST)}
        cleaned = " ".join(query.split())
        if cleaned:
            params["q"] = cleaned[:300]
        payload = await self._json(agent, "GET", "/messages", params=params)
        rows = payload.get("messages")
        if not isinstance(rows, list):
            return []
        found: list[ListedMail] = []
        for row in rows[:MAX_LIST]:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            found.append(await self._metadata(agent, row["id"]))
        return found

    async def read(self, agent: str, message_id: str) -> str:
        payload = await self._json(
            agent, "GET", f"/messages/{quote(message_id, safe='')}", params={"format": "full"}
        )
        meta = _listed(message_id, payload)
        body = _plain_text(payload.get("payload"))[:MAX_BODY]
        return f"From: {meta.sender}\nSubject: {meta.subject}\nDate: {meta.date}\n\n{body}"

    async def apply(self, agent: str, batch: FileBatch) -> str:
        """Check every preview, then change labels. A mismatch changes nothing."""
        live = [await self._metadata(agent, item.id) for item in batch.messages]
        mismatches = [
            item.subject
            for item, current in zip(batch.messages, live, strict=True)
            if current.subject != item.subject or current.sender != item.sender
        ]
        if mismatches:
            shown = ", ".join(mismatches[:5])
            raise MailError(f"those messages no longer match the preview: {shown}")
        add, remove = await self._label_change(agent, batch)
        for item in batch.messages:
            await self._json(
                agent,
                "POST",
                f"/messages/{quote(item.id, safe='')}/modify",
                json={"addLabelIds": add, "removeLabelIds": remove},
            )
        return f"{batch.action} applied to {len(batch.messages)}."

    async def draft(self, agent: str, message: DraftMessage) -> str:
        raw = _rfc822(message)
        await self._json(agent, "POST", "/drafts", json={"message": {"raw": raw}})
        return f"Draft saved in Gmail: {message.subject}"

    async def exchange(self, agent: str, code: str, *, redirect_uri: str) -> None:
        secret = _client_secret(self._settings)
        client_id = self._settings.gmail_client_id.strip()
        if not client_id or secret is None:
            raise MailError("gmail is not configured")
        body = {
            "code": code.strip(),
            "client_id": client_id,
            "client_secret": secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        await self._store_grant(agent, body)

    def authorization_url(self, *, redirect_uri: str) -> str:
        client_id = self._settings.gmail_client_id.strip()
        if not client_id:
            raise MailError("gmail is not configured")
        query = urlencode(
            {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": GMAIL_SCOPE,
                "access_type": "offline",
                "prompt": "consent",
            }
        )
        return f"{AUTH_URL}?{query}"

    async def _label_change(self, agent: str, batch: FileBatch) -> tuple[list[str], list[str]]:
        if batch.action == "archive":
            return [], ["INBOX"]
        if batch.action == "star":
            return ["STARRED"], []
        if batch.action == "unstar":
            return [], ["STARRED"]
        if batch.action == "trash":
            return ["TRASH"], ["INBOX"]
        label_id = await self._label_id(agent, batch.label)
        return [label_id], ["INBOX"]

    async def _label_id(self, agent: str, name: str) -> str:
        payload = await self._json(agent, "GET", "/labels")
        rows = payload.get("labels")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and str(row.get("name", "")).casefold() == name.casefold():
                    label_id = row.get("id")
                    if isinstance(label_id, str):
                        return label_id
        raise MailError(f"no label named {name}")

    async def _metadata(self, agent: str, message_id: str) -> ListedMail:
        payload = await self._json(
            agent,
            "GET",
            f"/messages/{quote(message_id, safe='')}",
            params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
        )
        return _listed(message_id, payload)

    async def _json(
        self,
        agent: str,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | list[str]] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _refuse_path(method, path)
        token = await self._access_token(agent)
        try:
            response = await self._http.request(
                method,
                f"{GMAIL_API}{path}",
                params=params,
                json=json,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            raise MailError("gmail is unavailable") from exc
        if response.status_code in {401, 403}:
            raise MailError("gmail login was rejected")
        if response.status_code >= 400:
            raise MailError("gmail is unavailable")
        if response.status_code == 204 or not response.content:
            return {}
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise MailError("gmail returned unreadable data") from exc
        if not isinstance(payload, dict):
            raise MailError("gmail returned unreadable data")
        return payload

    async def _access_token(self, agent: str) -> str:
        stored = self._store.gmail_token(agent)
        if stored is None:
            raise MailError("gmail is not signed in for this agent")
        refresh, access, expires_at = stored
        if access and expires_at > time.time() + 30:
            return access
        secret = _client_secret(self._settings)
        client_id = self._settings.gmail_client_id.strip()
        if not client_id or secret is None:
            raise MailError("gmail is not configured")
        await self._store_grant(
            agent,
            {
                "client_id": client_id,
                "client_secret": secret,
                "refresh_token": refresh,
                "grant_type": "refresh_token",
            },
            keep_refresh=refresh,
        )
        refreshed = self._store.gmail_token(agent)
        if refreshed is None or not refreshed[1]:
            raise MailError("gmail login was rejected")
        return refreshed[1]

    async def _store_grant(
        self,
        agent: str,
        form: dict[str, str],
        *,
        keep_refresh: str = "",
    ) -> None:
        try:
            response = await self._http.post(TOKEN_URL, data=form)
        except httpx.HTTPError as exc:
            raise MailError("gmail is unavailable") from exc
        if response.status_code >= 400:
            raise MailError("gmail login was rejected")
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise MailError("gmail login was rejected") from exc
        if not isinstance(payload, dict):
            raise MailError("gmail login was rejected")
        access = payload.get("access_token")
        refresh = payload.get("refresh_token")
        if not isinstance(access, str) or not access:
            raise MailError("gmail login was rejected")
        if not isinstance(refresh, str) or not refresh:
            refresh = keep_refresh
        if not refresh:
            raise MailError("gmail login was rejected")
        expires_in = payload.get("expires_in")
        seconds = float(expires_in) if isinstance(expires_in, (int, float)) else 3600.0
        self._store.save_gmail_token(
            agent,
            refresh_token=refresh,
            access_token=access,
            expires_at=time.time() + seconds,
        )


def _refuse_path(method: str, path: str) -> None:
    lowered = path.lower()
    if method.upper() == "DELETE" or lowered.endswith("/send") or "/delete" in lowered:
        raise MailError("gmail send and permanent delete are not available")


def _client_secret(settings: Settings) -> str | None:
    secret = settings.gmail_client_secret
    if secret is None:
        return None
    value = secret.get_secret_value().strip()
    return value or None


def _listed(message_id: str, payload: dict[str, Any]) -> ListedMail:
    headers = _headers(payload.get("payload"))
    snippet = payload.get("snippet")
    return ListedMail(
        id=message_id,
        sender=headers.get("from", ""),
        subject=headers.get("subject", "(no subject)"),
        date=headers.get("date", ""),
        snippet=snippet.strip()[:240] if isinstance(snippet, str) else "",
    )


def _headers(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("headers")
    found: dict[str, str] = {}
    if not isinstance(rows, list):
        return found
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        value = row.get("value")
        if isinstance(name, str) and name in _HEADER and isinstance(value, str):
            found[name.lower()] = " ".join(value.split())[:500]
    return found


def _plain_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    mime = payload.get("mimeType")
    if mime == "text/plain":
        return _decode_body(payload.get("body"))
    parts = payload.get("parts")
    if isinstance(parts, list):
        for part in parts:
            text = _plain_text(part)
            if text:
                return text
    return _decode_body(payload.get("body"))


def _decode_body(body: object) -> str:
    if not isinstance(body, dict):
        return ""
    data = body.get("data")
    if not isinstance(data, str) or not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")
    except (ValueError, UnicodeError):
        return ""


def _rfc822(message: DraftMessage) -> str:
    lines = [
        f"To: {message.to}",
        f"Subject: {message.subject}",
        "Content-Type: text/plain; charset=utf-8",
        "",
        message.body,
    ]
    raw = "\r\n".join(lines).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")
