from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pytest
from structlog.testing import capture_logs

import assistai.__main__ as cli
from assistai.compare import CompareResult
from assistai.errors import MissingAPIKeyError
from assistai.jobs import Job
from assistai.store import Store, store_path
from tests.clock_fakes import chicago
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


_MGMPLUS_PROMPT = "Cancel MGM+ before the rate goes up."


def _state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, timezone: bool = True) -> Path:
    """Point the store commands at a state directory, with a stopped clock and quiet logs."""
    state_dir = tmp_path / "state"
    monkeypatch.setenv("ASSISTAI_STATE_DIR", str(state_dir))
    if timezone:
        monkeypatch.setenv("ASSISTAI_TIMEZONE", "America/Chicago")
    monkeypatch.setattr(cli, "_wall_now", lambda: chicago(2026, 11, 9, 12))
    # The real one replaces structlog's global pipeline, which other tests capture.
    monkeypatch.setattr(cli, "configure_logging", lambda **_kwargs: None)
    return store_path(state_dir)


def _save(path: Path, name: str, prompt: str, *, cancelled: bool = False) -> None:
    """A daily 8:00 AM reminder first due Dec 10, as the Pi's mgmplus row is."""
    store = Store(path)
    first = chicago(2026, 12, 10, 8)
    store.save_job(
        Job(
            id="j00000002" if cancelled else "j00000001",
            agent="jacob",
            kind="schedule",
            name=name,
            prompt=prompt,
            every_seconds=86400,
            ttl_seconds=None,
            created_at=chicago(2026, 10, 30, 21, 12),
            expires_at=None,
            next_run_at=first,
            last_run_at=None,
            cancelled_at=chicago(2026, 10, 31) if cancelled else None,
            verbatim=True,
            zone=None if cancelled else "America/Chicago",
            anchor_at=None if cancelled else first,
            every_days=None if cancelled else 1,
            at_local=None if cancelled else "08:00",
        )
    )
    store.close()


def test_store_jobs_lists_names_and_times_and_no_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator's check before and after a deploy. A prompt can hold mail; it is never shown."""
    path = _state(tmp_path, monkeypatch)
    _save(path, "mgmplus cancel reminder", _MGMPLUS_PROMPT)
    _save(path, "morning email", "Summarize unread mail.", cancelled=True)

    with capture_logs() as logs:
        cli.main(["store", "jobs", "--next", "3"])

    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert "mgmplus cancel reminder" in lines[0]
    assert "every day at 8:00 AM" in lines[0]
    assert [line.strip() for line in lines[1:]] == [
        "Thursday, December 10 at 8:00 AM",
        "Friday, December 11 at 8:00 AM",
        "Saturday, December 12 at 8:00 AM",
    ]
    assert "morning email" not in out
    assert _MGMPLUS_PROMPT not in out + err + repr(logs)
    assert "Summarize" not in out + err + repr(logs)


def test_store_jobs_from_a_date_crosses_spring_forward_at_eight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The deploy B rehearsal: March 14 and 15, 2027 still show 8:00 AM."""
    path = _state(tmp_path, monkeypatch)
    _save(path, "mgmplus cancel reminder", _MGMPLUS_PROMPT)

    with capture_logs():
        cli.main(["store", "jobs", "--from", "2027-03-12", "--next", "4"])

    runs = [line.strip() for line in capsys.readouterr().out.splitlines()[1:]]
    assert runs == [
        "Friday, March 12, 2027 at 8:00 AM",
        "Saturday, March 13, 2027 at 8:00 AM",
        "Sunday, March 14, 2027 at 8:00 AM",
        "Monday, March 15, 2027 at 8:00 AM",
    ]


def test_store_jobs_fills_rows_an_older_build_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run on a copy of the Pi's file, it shows what the new image will run."""
    path = _state(tmp_path, monkeypatch)
    _save(path, "mgmplus cancel reminder", _MGMPLUS_PROMPT)
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE jobs SET zone = NULL, anchor_at = NULL, every_days = NULL, at_local = NULL"
    )
    conn.commit()
    conn.close()

    with capture_logs() as logs:
        cli.main(["store", "jobs", "--next", "1"])

    out = capsys.readouterr().out
    assert "every day at 8:00 AM" in out
    assert "Thursday, December 10 at 8:00 AM" in out
    assert [entry["event"] for entry in logs] == ["store.job_schedule_migrated"]


def test_store_jobs_needs_the_zone_and_an_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _state(tmp_path, monkeypatch, timezone=False)
    _save(path, "mgmplus cancel reminder", _MGMPLUS_PROMPT)
    with pytest.raises(SystemExit) as no_zone:
        cli.main(["store", "jobs"])
    assert no_zone.value.code == 2
    assert "ASSISTAI_TIMEZONE" in capsys.readouterr().err

    monkeypatch.setenv("ASSISTAI_TIMEZONE", "America/Chicago")
    monkeypatch.setenv("ASSISTAI_STATE_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(SystemExit) as no_file:
        cli.main(["store", "jobs"])
    assert no_file.value.code == 2
    assert "no state database" in capsys.readouterr().err
    assert not (tmp_path / "elsewhere").exists()


def test_store_downgrade_steps_a_recognised_file_back_to_14(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _state(tmp_path, monkeypatch)
    Store(path).close()

    cli.main(["store", "downgrade", "--to", "14"])

    assert "version 14" in capsys.readouterr().out
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 14
    conn.close()


@pytest.mark.parametrize("target", ["13", "15"])
def test_store_downgrade_refuses_another_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], target: str
) -> None:
    path = _state(tmp_path, monkeypatch)
    Store(path).close()

    with pytest.raises(SystemExit) as exc:
        cli.main(["store", "downgrade", "--to", target])

    assert exc.value.code == 2
    assert f"cannot downgrade to version {target}" in capsys.readouterr().err


def test_store_downgrade_refuses_a_changed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _state(tmp_path, monkeypatch)
    Store(path).close()
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE surprise (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    with pytest.raises(SystemExit) as exc:
        cli.main(["store", "downgrade", "--to", "14"])

    assert exc.value.code == 2
    assert "refusing to downgrade" in capsys.readouterr().err
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 15
    conn.close()
