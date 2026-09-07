"""Structured logging setup.

JSON by default so the Pi's journal stays greppable. Console rendering is opt-in
for local development.
"""

from __future__ import annotations

import logging

import structlog

from assistai.config import LogLevel


def configure_logging(level: LogLevel = "info", console: bool = False) -> None:
    """Install a structlog pipeline and align the stdlib root logger with it."""
    numeric_level = getattr(logging, level.upper())
    logging.basicConfig(format="%(message)s", level=numeric_level, force=True)
    # httpx logs every request at INFO; that is noise in the REPL and on the Pi.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer() if console else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
