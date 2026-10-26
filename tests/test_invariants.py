"""Repository rules that hold for every change, checked mechanically.

Each failure message says how to fix it, because the reader is as likely to be
a coding agent as a person. Loosening one of these is a design decision: record
it in docs/decisions/ before editing the expectation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from assistai.broker import ToolMeta, builtin_catalog

# Every tool the process can run and the broker attributes it was registered
# with. ToolCatalog.add defaults to trusted=True, so a tool added without
# thinking about taint would silently be trusted. This table forces the
# thought: a new tool fails here until someone classifies it on purpose.
EXPECTED_TOOLS: dict[str, ToolMeta] = {
    "calendar_add": ToolMeta(trusted=True, sink="calendar:write", staging=True),
    "calendar_today": ToolMeta(trusted=False),
    "get_time": ToolMeta(trusted=True),
    "job_cancel": ToolMeta(trusted=True, staging=True),
    "job_create": ToolMeta(trusted=True, staging=True),
    "job_list": ToolMeta(trusted=True),
    "job_reschedule": ToolMeta(trusted=True, staging=True),
    "mail_draft": ToolMeta(trusted=True, sink="gmail:draft", staging=True),
    "mail_file": ToolMeta(trusted=True, sink="gmail:file", staging=True),
    "mail_inbox": ToolMeta(trusted=False),
    "mail_read": ToolMeta(trusted=False),
    "relay": ToolMeta(trusted=True, sink="message:other_peer", staging=True),
    "shared_change": ToolMeta(trusted=True, sink="shared:write", staging=True),
    "shared_lists": ToolMeta(trusted=False),
    "web_fetch": ToolMeta(trusted=False, web=True),
    "web_search": ToolMeta(trusted=False, web=True),
}

# Packages that reach a provider or hold household data. They are called by the
# broker and gateway and must not reach back into routing, transport, or storage.
_DOMAINS = ("calendar", "mail", "research", "shared")
_ORCHESTRATION = ("broker", "gateway", "scheduler", "signal", "store")

# The model loop may not import anything that could let it run a tool without
# the broker in front of it.
FORBIDDEN_IMPORTS: dict[str, frozenset[str]] = {
    "inference": frozenset({*_ORCHESTRATION, *_DOMAINS, "jobs", "relay", "staging", "agents"}),
    **{domain: frozenset(_ORCHESTRATION) for domain in _DOMAINS},
}

_SRC = Path(__file__).resolve().parents[1] / "src" / "assistai"
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def test_every_tool_has_deliberate_security_metadata() -> None:
    catalog = builtin_catalog()
    actual = {name: catalog.meta(name) for name in catalog.names()}

    added = sorted(set(actual) - set(EXPECTED_TOOLS))
    removed = sorted(set(EXPECTED_TOOLS) - set(actual))
    assert not added, (
        f"new tools {added} are not classified. Decide trusted, sink, web, and staging, "
        "then add them to EXPECTED_TOOLS in tests/test_invariants.py and to the "
        "permissions table in docs/architecture.md."
    )
    assert not removed, f"tools {removed} were removed; drop them from EXPECTED_TOOLS."
    changed = {
        name: (EXPECTED_TOOLS[name], meta)
        for name, meta in actual.items()
        if meta != EXPECTED_TOOLS[name]
    }
    assert not changed, (
        f"security metadata changed (expected, actual): {changed}. This changes what "
        "taint and staging protect. Record why in docs/decisions/, then update the table."
    )


def test_every_sink_is_staged() -> None:
    catalog = builtin_catalog()
    unstaged = sorted(
        name
        for name in catalog.names()
        if catalog.meta(name).sink is not None and not catalog.meta(name).staging
    )
    assert not unstaged, (
        f"{unstaged} write outside the conversation without a yes. Mark them staging=True "
        "so the person confirms the stored call; see 'Staged actions' in docs/architecture.md."
    )


def test_untrusted_tools_are_not_sinks() -> None:
    catalog = builtin_catalog()
    both = sorted(
        name
        for name in catalog.names()
        if not catalog.meta(name).trusted and catalog.meta(name).sink is not None
    )
    assert not both, (
        f"{both} both ingest untrusted text and write. Split the read and the write into "
        "two tools so taint from the read can gate the write."
    )


def test_package_layering() -> None:
    violations: list[str] = []
    for package, forbidden in FORBIDDEN_IMPORTS.items():
        for path in sorted((_SRC / package).rglob("*.py")):
            for target in _assistai_imports(path):
                if target in forbidden:
                    violations.append(f"{path.relative_to(_SRC)} imports assistai.{target}")
    assert not violations, (
        "layering violations:\n  "
        + "\n  ".join(violations)
        + "\nDomain packages and the inference loop must not reach the broker, transport, "
        "or store. Pass what they need in as an argument or a Protocol, the way "
        "bind_mail takes a store."
    )


def test_relative_doc_links_resolve(repo_root: Path) -> None:
    pages = [
        repo_root / "README.md",
        repo_root / "AGENTS.md",
        *sorted((repo_root / "docs").rglob("*.md")),
        *sorted((repo_root / ".cursor").rglob("*.md")),
        *sorted((repo_root / ".cursor").rglob("*.mdc")),
    ]
    broken: list[str] = []
    for page in pages:
        if not page.is_file():
            continue
        for target in _LINK.findall(page.read_text(encoding="utf-8")):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            if not (page.parent / path).exists():
                broken.append(f"{page.relative_to(repo_root)} -> {target}")
    assert not broken, "broken relative links:\n  " + "\n  ".join(broken)


def test_the_import_walker_sees_every_import_form() -> None:
    """A form the walker skips is a layering violation nobody is told about."""
    source = "\n".join(
        [
            "import assistai.store",
            "from assistai.broker import ToolMeta",
            "from assistai import gateway",
            "from .. import scheduler",
            "from ..signal.channel import SignalChannel",
            "from .types import Message",
        ]
    )

    found = _imports_in(source, package=("assistai", "inference"))

    assert found == {"store", "broker", "gateway", "scheduler", "signal", "inference"}


def _assistai_imports(path: Path) -> set[str]:
    package = ("assistai", *path.relative_to(_SRC).parent.parts)
    return _imports_in(path.read_text(encoding="utf-8"), package=package)


def _imports_in(source: str, *, package: tuple[str, ...]) -> set[str]:
    """Top-level assistai subpackages a module imports, in any import form."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        targets: list[list[str]] = []
        if isinstance(node, ast.Import):
            targets = [alias.name.split(".") for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = list(package[: len(package) - node.level + 1]) if node.level else []
            module = [*base, *(node.module.split(".") if node.module else [])]
            if len(module) == 1:
                # `from assistai import broker` and `from .. import store` name
                # the subpackage in the alias, not the module.
                targets = [[*module, alias.name] for alias in node.names]
            else:
                targets = [module]
        for parts in targets:
            if len(parts) > 1 and parts[0] == "assistai":
                found.add(parts[1])
    return found
