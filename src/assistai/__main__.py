"""Process entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys

from assistai.chat import chat_once, chat_repl
from assistai.compare import compare_models, format_report
from assistai.config import Settings
from assistai.errors import AssistAIError
from assistai.gateway import Gateway, install_signal_handlers
from assistai.logging import configure_logging
from assistai.manifest import load_manifest, resolve_manifest_path
from assistai.signal.client import SignalClient
from assistai.signal.numbers import normalize_e164


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

    signal = sub.add_parser("signal", help="signal-cli setup and diagnostics")
    signal_sub = signal.add_subparsers(dest="signal_command", required=True)
    signal_sub.add_parser("health", help="check signal-cli and list accounts")
    signal_sub.add_parser("link", help="print a device-link URI to scan from Signal")
    register = signal_sub.add_parser("register", help="start SMS/voice registration")
    register.add_argument("number", help="dedicated bot number in E.164")
    register.add_argument("--captcha", default=None, help="token from signalcaptchas.org")
    register.add_argument("--voice", action="store_true", help="call instead of SMS")
    verify = signal_sub.add_parser("verify", help="complete registration with the SMS code")
    verify.add_argument("number", help="dedicated bot number in E.164")
    verify.add_argument("code", help="verification code")

    models = sub.add_parser("models", help="compare pinned models against real tool schemas")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("compare", help="score primary and candidates on get_time")

    args = parser.parse_args(argv)
    try:
        if args.command == "chat":
            asyncio.run(_chat(args.once, use_tools=not args.no_tools))
        elif args.command == "signal":
            asyncio.run(_signal(args))
        elif args.command == "models":
            asyncio.run(_models(args))
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


async def _signal(args: argparse.Namespace) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, console=True)
    client = SignalClient(settings)
    try:
        if args.signal_command == "health":
            await client.check()
            accounts = await client.accounts()
            print("signal-cli ok")
            if accounts:
                print("accounts: " + ", ".join(accounts))
            else:
                print("accounts: (none registered)")
            return
        if args.signal_command == "link":
            uri = await client.link_uri(settings.signal_device_name)
            print(uri)
            print(
                "Scan from Signal: Settings → Linked devices → Link new device.",
                file=sys.stderr,
            )
            return
        if args.signal_command == "register":
            number = normalize_e164(args.number)
            await client.register(number, captcha=args.captcha, voice=args.voice)
            print(f"verification code sent to {number}")
            return
        if args.signal_command == "verify":
            number = normalize_e164(args.number)
            await client.verify(number, args.code)
            print(f"registered {number}")
            return
    finally:
        await client.aclose()


async def _models(args: argparse.Namespace) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, console=True)
    if args.models_command != "compare":
        return
    manifest = load_manifest(resolve_manifest_path(settings))
    results = await compare_models(settings, manifest)
    print(format_report(results))
    if not any(row.available and row.tool_called for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
