from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from assistai.broker import ToolBroker, builtin_catalog
from assistai.errors import MailError
from assistai.inference.types import ToolCall
from assistai.mail.actions import parse_file
from assistai.mail.client import GmailClient
from assistai.mail.scope import GMAIL_SCOPE, assert_scope_allowed
from assistai.mail.tools import MAIL_FILE, MAIL_INBOX
from assistai.staging import format_proposal
from tests.agent_fakes import agent, household
from tests.fakes import settings, tool_call

_SECRET = "client-secret"
_REFRESH = "refresh-token"
_ACCESS = "access-token"


class _Tokens:
    def __init__(self, *, signed_in: set[str] | None = None) -> None:
        self.saved: list[str] = []
        self._signed_in = signed_in or set()

    def gmail_token(self, agent: str) -> tuple[str, str, float] | None:
        if agent not in self._signed_in:
            return None
        return _REFRESH, _ACCESS, 10**12

    def save_gmail_token(
        self,
        agent: str,
        *,
        refresh_token: str,
        access_token: str,
        expires_at: float,
    ) -> None:
        self.saved.append(agent)
        self._signed_in.add(agent)
        assert refresh_token
        assert access_token
        assert expires_at > 0


def _configured() -> Any:
    return settings(gmail_client_id="client", gmail_client_secret=_SECRET)


def test_scope_is_modify_and_not_send() -> None:
    assert GMAIL_SCOPE.endswith("/gmail.modify")
    assert "send" not in GMAIL_SCOPE
    assert "compose" not in GMAIL_SCOPE
    assert_scope_allowed(GMAIL_SCOPE)


async def test_authorization_url_requests_only_modify() -> None:
    client = GmailClient(_configured(), _Tokens())
    url = client.authorization_url(redirect_uri="http://127.0.0.1:8731/")
    await client.aclose()
    assert "gmail.modify" in url
    assert "gmail.send" not in url
    assert "gmail.compose" not in url


async def test_inbox_uses_that_agents_token_and_not_send() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert _SECRET not in str(request.url)
        assert "/send" not in request.url.path
        assert request.method != "DELETE"
        assert request.headers["authorization"] == f"Bearer {_ACCESS}"
        if request.url.path.endswith("/messages/abc"):
            return httpx.Response(
                200,
                json={
                    "id": "abc",
                    "snippet": "hello",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "Ada <ada@example.com>"},
                            {"name": "Subject", "value": "Dentist"},
                            {"name": "Date", "value": "Tue, 22 Sep 2026"},
                        ]
                    },
                },
            )
        return httpx.Response(200, json={"messages": [{"id": "abc"}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GmailClient(_configured(), _Tokens(signed_in={"jacob"}), http=http)
    rows = await client.inbox("jacob")
    await http.aclose()
    assert rows[0].subject == "Dentist"
    assert rows[0].sender == "Ada <ada@example.com>"
    assert seen


async def test_other_agent_is_not_signed_in() -> None:
    client = GmailClient(_configured(), _Tokens(signed_in={"jacob"}))
    with pytest.raises(MailError, match="not signed in"):
        await client.inbox("spouse")
    await client.aclose()


async def test_archive_checks_the_preview_then_removes_inbox() -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method + " " + request.url.path, body))
        assert "/delete" not in request.url.path
        if request.url.path.endswith("/modify"):
            return httpx.Response(200, json={})
        return httpx.Response(
            200,
            json={
                "payload": {
                    "headers": [
                        {"name": "From", "value": "News <news@example.com>"},
                        {"name": "Subject", "value": "Weekly"},
                    ]
                }
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GmailClient(_configured(), _Tokens(signed_in={"jacob"}), http=http)
    batch = parse_file(
        {
            "action": "archive",
            "messages": [{"id": "abc", "subject": "Weekly", "from": "News <news@example.com>"}],
        }
    )
    message = await client.apply("jacob", batch)
    await http.aclose()
    assert message.startswith("archive applied")
    modify = next(body for path, body in calls if path.endswith("/modify"))
    assert modify == {"addLabelIds": [], "removeLabelIds": ["INBOX"]}


async def test_a_changed_subject_files_nothing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert not request.url.path.endswith("/modify")
        return httpx.Response(
            200,
            json={
                "payload": {
                    "headers": [
                        {"name": "From", "value": "News <news@example.com>"},
                        {"name": "Subject", "value": "Something else"},
                    ]
                }
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GmailClient(_configured(), _Tokens(signed_in={"jacob"}), http=http)
    batch = parse_file(
        {
            "action": "trash",
            "messages": [{"id": "abc", "subject": "Weekly", "from": "News <news@example.com>"}],
        }
    )
    with pytest.raises(MailError, match="no longer match"):
        await client.apply("jacob", batch)
    await http.aclose()


async def test_mail_file_stages_and_inbox_taints() -> None:
    jacob = agent("jacob", "+15555550101", tools=(MAIL_INBOX, MAIL_FILE))
    catalog = builtin_catalog()

    async def inbox(_arguments: dict[str, Any]) -> str:
        return "[]"

    catalog.bind(MAIL_INBOX, inbox)
    surface = ToolBroker(
        catalog, household(jacob, agent("spouse", "+15555550102")).broker
    ).for_agent(jacob)
    await surface.execute(tool_call(name=MAIL_INBOX, arguments="{}"))
    assert surface.tainted is True
    staged = await surface.execute(
        tool_call(
            name=MAIL_FILE,
            arguments=json.dumps(
                {
                    "action": "archive",
                    "messages": [{"id": "abc", "subject": "Weekly", "from": "News"}],
                }
            ),
        )
    )
    assert "staged" in staged.content
    report = ToolBroker(
        catalog, household(jacob, agent("spouse", "+15555550102")).broker
    ).for_agent(jacob, report_only=True)
    assert MAIL_INBOX in {spec.name for spec in report.specs()}
    assert MAIL_FILE not in {spec.name for spec in report.specs()}


def test_file_preview_lists_every_message() -> None:
    preview = format_proposal(
        (
            ToolCall(
                id="c1",
                name="mail_file",
                arguments=json.dumps(
                    {
                        "action": "archive",
                        "messages": [
                            {"id": "abc", "subject": "Weekly", "from": "News"},
                            {"id": "def", "subject": "Receipt", "from": "Shop"},
                        ],
                    }
                ),
            ),
        ),
        tainted=True,
    )
    assert "Weekly" in preview
    assert "Receipt" in preview
    assert "untrusted" in preview
    assert "yes" in preview.lower()
