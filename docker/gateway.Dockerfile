# syntax=docker/dockerfile:1
#
# Builds for linux/amd64 and linux/arm64 from the same definition, so the image
# developed on the Mac is the image that runs on the Pi.
#
# Requires uv.lock to exist: run `make lock` first. The build uses --locked and
# fails rather than silently resolving new versions.

FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve in their own layer so source edits do not invalidate them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY manifest.toml ./
RUN uv sync --locked --no-dev


FROM python:3.12-slim-bookworm

# No shell tooling, no package manager use at runtime, never root.
RUN useradd --system --create-home --uid 10001 assistai

WORKDIR /app

COPY --from=builder --chown=assistai:assistai /app/.venv /app/.venv
COPY --from=builder --chown=assistai:assistai /app/src /app/src
COPY --from=builder --chown=assistai:assistai /app/manifest.toml /app/manifest.toml

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ASSISTAI_MANIFEST_PATH=/app/manifest.toml \
    ASSISTAI_STATE_DIR=/app/state \
    ASSISTAI_SIGNAL_BASE_URL=http://signal-cli:8080

USER assistai

ENTRYPOINT ["python", "-m", "assistai"]
