"""``python -m evals``: run the golden cases and exit non-zero on a regression."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from assistai.config import Settings
from assistai.logging import configure_logging
from assistai.manifest import ModelPin, load_manifest, resolve_manifest_path
from evals.harness import CASES_DIR, CaseResult, load_cases, metered, run_case

_REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals")
    parser.add_argument("--trials", type=int, default=3, help="runs per case (default 3)")
    parser.add_argument("--case", action="append", default=[], help="only this case id")
    parser.add_argument("--tag", action="append", default=[], help="only cases with this tag")
    parser.add_argument(
        "--model", default="primary", help="primary, or a name under models.candidates"
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=_REPO / "config" / "assistai.example.toml",
        help="roster whose first agent supplies the default tool list",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    sys.exit(asyncio.run(_run(args)))


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    configure_logging(level="warning", console=True)
    manifest = load_manifest(resolve_manifest_path(settings))
    pin = _pin(manifest.primary, manifest.candidates, args.model)
    cases = load_cases(CASES_DIR, roster=args.roster)
    if args.case:
        cases = [case for case in cases if case.id in args.case]
    if args.tag:
        cases = [case for case in cases if set(case.tags) & set(args.tag)]
    if not cases:
        print("no cases selected")
        return 2
    client = metered(settings)
    results: list[CaseResult] = []
    try:
        for case in cases:
            result = await run_case(
                client, pin, case, trials=args.trials, max_tool_rounds=settings.max_tool_rounds
            )
            results.append(result)
            _print(result)
    finally:
        await client.aclose()
    failed = [result for result in results if not result.passed]
    prompt = sum(trial.prompt_tokens for result in results for trial in result.trials)
    completion = sum(trial.completion_tokens for result in results for trial in result.trials)
    print(
        f"\n{len(results) - len(failed)}/{len(results)} cases passed on {pin.ref} "
        f"({prompt} prompt + {completion} completion tokens)"
    )
    return 1 if failed else 0


def _pin(primary: ModelPin, candidates: dict[str, ModelPin], name: str) -> ModelPin:
    if name == "primary":
        return primary
    if name not in candidates:
        raise SystemExit(f"unknown model {name!r}; choose primary or one of {sorted(candidates)}")
    return candidates[name]


def _print(result: CaseResult) -> None:
    mark = "PASS" if result.passed else "FAIL"
    print(f"{mark} {result.case.id:<28} {result.passes}/{len(result.trials)}")
    if result.passed:
        return
    for index, trial in enumerate(result.trials, start=1):
        if trial.passed:
            continue
        print(f"     trial {index}: {'; '.join(trial.failures)}")
        print(f"       calls={list(trial.calls)} staged={list(trial.staged)}")
        print(f"       reply={trial.reply[:160]!r}")


if __name__ == "__main__":
    main()
