from __future__ import annotations

from pathlib import Path

import pytest

from assistai.config import Settings
from assistai.errors import ManifestError
from assistai.manifest import load_manifest, resolve_manifest_path


def test_loads_repo_manifest(repo_root: Path) -> None:
    manifest = load_manifest(repo_root / "manifest.toml")

    assert manifest.primary.ref == "accounts/fireworks/models/minimax-m3"
    assert manifest.primary.thinking == "disabled"
    assert manifest.quarantine.ref == "accounts/fireworks/models/deepseek-v4-flash-0731"
    assert "glm_5p3" in manifest.candidates
    assert manifest.required_refs() == (
        manifest.primary.ref,
        manifest.quarantine.ref,
    )


def test_swap_is_one_field(tmp_path: Path) -> None:
    path = tmp_path / "manifest.toml"
    path.write_text(
        """
[models.primary]
provider = "fireworks"
ref = "accounts/fireworks/models/glm-5p3"

[models.quarantine]
provider = "fireworks"
ref = "accounts/fireworks/models/deepseek-v4-flash"
""",
        encoding="utf-8",
    )

    assert load_manifest(path).primary.ref.endswith("glm-5p3")


def test_missing_file() -> None:
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(Path("/no/such/manifest.toml"))


def test_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "manifest.toml"
    path.write_text("this is not = toml [", encoding="utf-8")

    with pytest.raises(ManifestError, match="unreadable"):
        load_manifest(path)


def test_missing_primary(tmp_path: Path) -> None:
    path = tmp_path / "manifest.toml"
    path.write_text("[models.quarantine]\nref = 'x'\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="primary"):
        load_manifest(path)


def test_empty_ref_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.toml"
    path.write_text(
        """
[models.primary]
provider = "fireworks"
ref = "   "

[models.quarantine]
provider = "fireworks"
ref = "accounts/fireworks/models/deepseek-v4-flash"
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="ref"):
        load_manifest(path)


def test_resolve_uses_explicit_setting(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    path.write_text("", encoding="utf-8")

    resolved = resolve_manifest_path(Settings(manifest_path=path))

    assert resolved == path


def test_resolve_uses_cwd_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "manifest.toml"
    path.write_text("[models]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_manifest_path(Settings()) == path


def test_resolve_fails_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ManifestError, match="ASSISTAI_MANIFEST_PATH"):
        resolve_manifest_path(Settings())
