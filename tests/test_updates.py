"""The update watcher flags a stale signal-cli and does not change anything."""

from __future__ import annotations

from pathlib import Path

from assistai.errors import UpdateError
from assistai.store import Store
from assistai.updates.check import PublishedTag, TagPage
from assistai.updates.hub import _next_page
from assistai.updates.watcher import UpdateWatcher
from tests.fakes import settings
from tests.signal_fakes import FakeSignal

_MANIFEST = """
[models.primary]
provider = "fireworks"
ref = "accounts/fireworks/models/minimax-m3"

[models.quarantine]
provider = "fireworks"
ref = "accounts/fireworks/models/deepseek-v4-flash-0731"

[images.signal_cli]
ref = "bbernhard/signal-cli-rest-api:0.100"
digest = ""
priority = "high"

[images.gateway_base]
ref = "python:3.12-slim-bookworm"
digest = ""
"""


class _Hub:
    def __init__(self, pages: dict[str, TagPage]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    async def tags(self, repository: str) -> TagPage:
        self.calls.append(repository)
        page = self.pages.get(repository)
        if page is None:
            return TagPage(tags=(), complete=True)
        return page

    async def aclose(self) -> None:
        return None


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.toml"
    path.write_text(_MANIFEST, encoding="utf-8")
    return path


def _watcher(tmp_path: Path, hub: _Hub, signal: FakeSignal) -> UpdateWatcher:
    path = _write_manifest(tmp_path)
    return UpdateWatcher(
        settings(
            manifest_path=path,
            signal_allow_from=("+15555550101", "+15555550102"),
        ),
        Store(tmp_path / "assistai.sqlite"),
        signal,
        source=hub,
    )


def _pages(signal: tuple[PublishedTag, ...]) -> dict[str, TagPage]:
    return {
        "bbernhard/signal-cli-rest-api": TagPage(tags=signal, complete=True),
        "library/python": TagPage(
            tags=(
                PublishedTag("3.12-slim-bookworm"),
                PublishedTag("3.13-slim-bookworm"),
                PublishedTag("latest"),
            ),
            complete=True,
        ),
    }


async def test_a_newer_signal_cli_is_flagged_and_nothing_is_changed(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    before = manifest.read_bytes()
    signal = FakeSignal()
    hub = _Hub(
        _pages(
            (
                PublishedTag("0.100"),
                PublishedTag("0.101"),
                PublishedTag("latest"),
            )
        )
    )
    watcher = UpdateWatcher(
        settings(
            manifest_path=manifest,
            signal_allow_from=("+15555550101", "+15555550102"),
        ),
        Store(tmp_path / "assistai.sqlite"),
        signal,
        source=hub,
    )

    report = await watcher.check()

    assert "signal-cli" in report
    assert "0.101" in report
    assert "high priority" in report
    assert "Nothing was installed or changed." in report
    assert "gateway_base" not in report
    assert signal.sent == [("+15555550101", report)]
    assert manifest.read_bytes() == before
    assert await watcher.check() == ""
    assert signal.sent == [("+15555550101", report)]


async def test_the_current_signal_cli_pin_is_quiet(tmp_path: Path) -> None:
    signal = FakeSignal()
    hub = _Hub(_pages((PublishedTag("0.100"), PublishedTag("latest"))))
    watcher = _watcher(tmp_path, hub, signal)

    assert await watcher.check() == ""
    assert signal.sent == []


async def test_a_failed_lookup_does_not_clear_a_stale_notice(tmp_path: Path) -> None:
    signal = FakeSignal()
    store = Store(tmp_path / "assistai.sqlite")
    store.save_update_report("signal-cli is stale", now=1.0)
    hub = _Hub({})

    async def tags(_repository: str) -> TagPage:
        raise UpdateError("Docker Hub is unreachable")

    hub.tags = tags  # type: ignore[method-assign]
    watcher = UpdateWatcher(
        settings(
            manifest_path=_write_manifest(tmp_path),
            signal_allow_from=("+15555550101",),
        ),
        store,
        signal,
        source=hub,
    )

    assert await watcher.check() == ""
    assert signal.sent == []
    assert store.last_update_report() == "signal-cli is stale"
    store.close()


async def test_a_confirmed_stale_pin_is_sent_when_another_lookup_fails(tmp_path: Path) -> None:
    signal = FakeSignal()
    store = Store(tmp_path / "assistai.sqlite")
    pages = _pages((PublishedTag("0.100"), PublishedTag("0.102"), PublishedTag("latest")))

    class _Flaky(_Hub):
        async def tags(self, repository: str) -> TagPage:
            if repository == "library/python":
                raise UpdateError("Docker Hub is unreachable")
            return await super().tags(repository)

    watcher = UpdateWatcher(
        settings(
            manifest_path=_write_manifest(tmp_path),
            signal_allow_from=("+15555550101",),
        ),
        store,
        signal,
        source=_Flaky(pages),
    )

    report = await watcher.check()
    assert "0.102" in report
    assert "gateway_base" not in report
    assert await watcher.check() == ""
    assert len(signal.sent) == 1
    store.close()


def test_a_truncated_list_still_flags_a_newer_release() -> None:
    from assistai.manifest import ImagePin
    from assistai.updates.check import assess

    pin = ImagePin(
        name="signal_cli",
        ref="bbernhard/signal-cli-rest-api:0.100",
        repository="bbernhard/signal-cli-rest-api",
        tag="0.100",
        digest="",
        priority="high",
    )
    found = assess(pin, [PublishedTag("0.101"), PublishedTag("latest")], complete=False)
    assert found is not None
    assert "0.101" in found.detail


def test_a_tag_page_on_another_host_is_refused() -> None:
    import pytest

    from assistai.errors import UpdateError

    with pytest.raises(UpdateError, match="another host"):
        _next_page(
            "https://evil.example/v2/repositories/bbernhard/signal-cli-rest-api/tags",
            "bbernhard/signal-cli-rest-api",
        )
    with pytest.raises(UpdateError, match="unexpected tag page"):
        _next_page(
            "https://hub.docker.com/v2/repositories/other/repo/tags?page=2",
            "bbernhard/signal-cli-rest-api",
        )
