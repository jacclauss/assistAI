from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import assistai.__main__ as cli
from assistai.compare import CompareResult
from assistai.errors import MissingAPIKeyError
from tests.fakes import manifest


def test_unknown_command_starts_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    async def fake_gateway() -> None:
        called.append("gateway")

    monkeypatch.setattr(cli, "_gateway", fake_gateway)
    cli.main([])

    assert called == ["gateway"]


def test_chat_once_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str | None, bool]] = []

    async def fake_chat(once: str | None, *, use_tools: bool) -> None:
        seen.append((once, use_tools))

    monkeypatch.setattr(cli, "_chat", fake_chat)
    cli.main(["chat", "--once", "ping", "--no-tools"])

    assert seen == [("ping", False)]


def test_extract_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    async def fake_extract() -> None:
        called.append("extract")

    monkeypatch.setattr(cli, "_extract", fake_extract)
    cli.main(["extract"])

    assert called == ["extract"]


def test_models_compare_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def fake_models(args: argparse.Namespace) -> None:
        seen.append(args.models_command)

    monkeypatch.setattr(cli, "_models", fake_models)
    cli.main(["models", "compare"])

    assert seen == ["compare"]


def test_assistai_error_exits_two(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom() -> None:
        raise MissingAPIKeyError("no key")

    monkeypatch.setattr(cli, "_gateway", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main([])

    assert exc.value.code == 2


class _StubSignal:
    """Records what the operator commands ask signal-cli to do."""

    def __init__(self, _settings: object, accounts: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[str, object]] = []
        self._accounts = accounts
        self.closed = False

    async def check(self) -> None:
        self.calls.append(("check", None))

    async def accounts(self) -> list[str]:
        self.calls.append(("accounts", None))
        return list(self._accounts)

    async def link_uri(self, device_name: str) -> str:
        self.calls.append(("link_uri", device_name))
        return "sgnl://linkdevice?uuid=abc"

    async def register(self, number: str, *, captcha: str | None, voice: bool) -> None:
        self.calls.append(("register", (number, captcha, voice)))

    async def verify(self, number: str, code: str) -> None:
        self.calls.append(("verify", (number, code)))

    async def lift_rate_limit(self, *, challenge_token: str, captcha: str) -> None:
        self.calls.append(("challenge", (challenge_token, captcha)))

    async def aclose(self) -> None:
        self.closed = True


def _stub_signal(
    monkeypatch: pytest.MonkeyPatch, accounts: tuple[str, ...] = ()
) -> list[_StubSignal]:
    made: list[_StubSignal] = []

    def factory(settings: object) -> _StubSignal:
        client = _StubSignal(settings, accounts=accounts)
        made.append(client)
        return client

    monkeypatch.setattr(cli, "SignalClient", factory)
    return made


def test_signal_health_reports_registered_accounts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made = _stub_signal(monkeypatch, accounts=("+15555550100",))

    cli.main(["signal", "health"])

    out = capsys.readouterr().out
    assert "signal-cli ok" in out
    assert "+15555550100" in out
    assert made[0].closed is True


def test_signal_health_says_none_when_unregistered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The empty case is the one an operator actually hits first."""
    _stub_signal(monkeypatch)

    cli.main(["signal", "health"])

    assert "(none registered)" in capsys.readouterr().out


def test_signal_link_prints_uri_on_stdout_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The URI has to be pipeable into a QR encoder, so hints go to stderr."""
    _stub_signal(monkeypatch)

    cli.main(["signal", "link"])

    captured = capsys.readouterr()
    assert captured.out.strip() == "sgnl://linkdevice?uuid=abc"
    assert "Linked devices" in captured.err


def test_signal_register_normalizes_the_number(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made = _stub_signal(monkeypatch)

    cli.main(["signal", "register", "+1 (555) 555-0100", "--captcha", "tok", "--voice"])

    assert made[0].calls == [("register", ("+15555550100", "tok", True))]
    assert "+15555550100" in capsys.readouterr().out


def test_signal_register_rejects_a_bad_number(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail before the call, not with an opaque REST error."""
    made = _stub_signal(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.main(["signal", "register", "555-0100"])

    assert exc.value.code == 2
    assert made[0].calls == []
    assert made[0].closed is True


def test_signal_verify_passes_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    made = _stub_signal(monkeypatch)

    cli.main(["signal", "verify", "+15555550100", "123456"])

    assert made[0].calls == [("verify", ("+15555550100", "123456"))]


def test_signal_challenge_posts_token_and_captcha(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made = _stub_signal(monkeypatch)

    cli.main(
        [
            "signal",
            "challenge",
            "3472e52f-7416-4e1a-8da3-668dfb59557c",
            "--captcha",
            "signalcaptcha://proof",
        ]
    )

    assert made[0].calls == [
        ("challenge", ("3472e52f-7416-4e1a-8da3-668dfb59557c", "signalcaptcha://proof"))
    ]
    assert "accepted" in capsys.readouterr().out


def test_models_compare_exits_nonzero_when_nothing_works(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bake-off where every model failed must not look like success in CI."""
    results = [
        CompareResult(
            name="primary",
            ref="accounts/fireworks/models/x",
            available=False,
            tool_called=False,
            arguments_valid=False,
            finished=False,
            error="InferenceError",
            latency_ms=1.0,
        )
    ]

    async def fake_compare(*_args: object, **_kwargs: object) -> list[CompareResult]:
        return results

    monkeypatch.setattr(cli, "load_manifest", lambda _path: manifest())
    monkeypatch.setattr(cli, "resolve_manifest_path", lambda _settings: Path("manifest.toml"))
    monkeypatch.setattr(cli, "compare_models", fake_compare)

    with pytest.raises(SystemExit) as exc:
        cli.main(["models", "compare"])

    assert exc.value.code == 1
    assert "primary" in capsys.readouterr().out


def test_models_compare_exits_zero_when_one_model_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        CompareResult(
            name="primary",
            ref="accounts/fireworks/models/x",
            available=True,
            tool_called=True,
            arguments_valid=True,
            finished=True,
            error=None,
            latency_ms=1.0,
        )
    ]

    async def fake_compare(*_args: object, **_kwargs: object) -> list[CompareResult]:
        return results

    monkeypatch.setattr(cli, "load_manifest", lambda _path: manifest())
    monkeypatch.setattr(cli, "resolve_manifest_path", lambda _settings: Path("manifest.toml"))
    monkeypatch.setattr(cli, "compare_models", fake_compare)

    cli.main(["models", "compare"])
