"""Load pinned model refs from ``manifest.toml``.

Swapping the primary model is a one-line change in that file. The client
sends ``ref`` to Fireworks as-is.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assistai.config import Settings
from assistai.errors import ManifestError


@dataclass(frozen=True)
class ModelPin:
    """A single named model pin from the manifest."""

    name: str
    provider: str
    ref: str
    thinking: str | None = None


@dataclass(frozen=True)
class Manifest:
    """The subset of the manifest that phase 1 needs."""

    primary: ModelPin
    quarantine: ModelPin
    candidates: dict[str, ModelPin]

    def required_refs(self) -> tuple[str, ...]:
        """Refs that must be callable at startup: primary and quarantine."""
        return (self.primary.ref, self.quarantine.ref)


def resolve_manifest_path(settings: Settings) -> Path:
    """Find ``manifest.toml`` without relying on the install layout.

    Installed wheels live under ``site-packages``, so a path relative to this
    file is not the repo root. Order: explicit setting, cwd, container default.
    """
    if settings.manifest_path is not None:
        return settings.manifest_path
    cwd = Path.cwd() / "manifest.toml"
    if cwd.is_file():
        return cwd
    container = Path("/app/manifest.toml")
    if container.is_file():
        return container
    raise ManifestError(
        "manifest.toml not found in the current directory; set ASSISTAI_MANIFEST_PATH"
    )


def load_manifest(path: Path) -> Manifest:
    """Parse and validate the model pins in ``path``."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {path}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"manifest unreadable: {path}") from exc

    models = raw.get("models")
    if not isinstance(models, dict):
        raise ManifestError("manifest is missing a [models] table")

    primary = _required_pin(models, "primary")
    quarantine = _required_pin(models, "quarantine")
    candidates = _candidate_pins(models.get("candidates"))
    return Manifest(primary=primary, quarantine=quarantine, candidates=candidates)


def _required_pin(models: dict[str, Any], name: str) -> ModelPin:
    block = models.get(name)
    if not isinstance(block, dict):
        raise ManifestError(f"manifest is missing [models.{name}]")
    ref = block.get("ref")
    provider = block.get("provider", "fireworks")
    if not isinstance(ref, str) or not ref.strip():
        raise ManifestError(f"models.{name}.ref must be a non-empty string")
    if not isinstance(provider, str) or not provider.strip():
        raise ManifestError(f"models.{name}.provider must be a non-empty string")
    thinking = block.get("thinking")
    if thinking is not None and not isinstance(thinking, str):
        raise ManifestError(f"models.{name}.thinking must be a string if set")
    return ModelPin(name=name, provider=provider, ref=ref.strip(), thinking=thinking)


def _candidate_pins(block: object) -> dict[str, ModelPin]:
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ManifestError("models.candidates must be a table")
    pins: dict[str, ModelPin] = {}
    for name, entry in block.items():
        if not isinstance(entry, dict):
            raise ManifestError(f"models.candidates.{name} must be a table")
        ref = entry.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            raise ManifestError(f"models.candidates.{name}.ref must be a non-empty string")
        pins[name] = ModelPin(name=name, provider="fireworks", ref=ref.strip())
    return pins
