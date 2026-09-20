"""Typed failures. Messages must never include secrets."""

from __future__ import annotations


class AssistAIError(Exception):
    """Base error for operator-facing failures."""


class MissingAPIKeyError(AssistAIError):
    """Fireworks credential is required for this command and was not set."""


class ManifestError(AssistAIError):
    """manifest.toml is missing, unreadable, or missing required pins."""


class ModelNotAvailableError(AssistAIError):
    """A pinned model ref is not callable on this Fireworks account."""


class InferenceError(AssistAIError):
    """The inference provider rejected or failed a request."""


class ToolLoopError(AssistAIError):
    """The agent loop could not complete a turn (malformed tools, round cap)."""


class SignalError(AssistAIError):
    """The signal-cli REST API rejected a request or is misconfigured."""


class SignalUnavailableError(SignalError):
    """signal-cli did not respond, or the configured account is not registered."""


class HouseholdConfigError(AssistAIError):
    """config/assistai.toml is missing, unreadable, or internally inconsistent."""


class StoreError(AssistAIError):
    """The state database is missing, unreadable, or newer than this build."""


class JobError(AssistAIError):
    """A job could not be created, changed, or run."""


class ResearchError(AssistAIError):
    """Search or fetch could not complete. Messages must not include secrets."""
