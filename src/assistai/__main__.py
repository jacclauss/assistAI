"""Process entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys

from assistai.chat import chat_once, chat_repl
from assistai.config import Settings
from assistai.errors import AssistAIError
from assistai.gateway import Gateway, install_signal_handlers
from assistai.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="assistai")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="multi-turn terminal conversation")
    chat.add_argument(
        "--once",
        metavar="TEXT",
        help="send one message and exit (for scripts and smoke tests)",
    )
    chat.add_argument(
        "--no-tools",
        action="store_true",
        help="do not advertise get_time; text-only completion",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "chat":
            asyncio.run(_chat(args.once, use_tools=not args.no_tools))
        else:
            asyncio.run(_gateway())
    except AssistAIError as exc:
        print(f"assistai: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


async def _gateway() -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, console=settings.log_console)
    gateway = Gateway(settings)
    install_signal_handlers(gateway)
    await gateway.run()


async def _chat(once: str | None, *, use_tools: bool) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, console=True)
    if once is not None:
        await chat_once(settings, once, use_tools=use_tools)
        return
    await chat_repl(settings, use_tools=use_tools)


if __name__ == "__main__":
    main()
