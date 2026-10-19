---
name: doc-garden
description: Finds places where AssistAI docs no longer match the code and proposes small corrections. Use when the user asks to check docs for drift, after a phase lands, or as a recurring maintenance task.
disable-model-invocation: true
---

# Doc garden

Goal: docs describe what the code does today. Small, separate fixes, each
reviewable in under a minute.

1. List claims in `README.md` (status table, setup steps, tool names),
   `docs/architecture.md` (permissions table, persistence list, phase table),
   `docs/release-notes.md`, and `AGENTS.md` (commands, layout, skills).
2. Verify each against the code. Useful checks:
   - Tool names and classification: `builtin_catalog()` in `broker.py` and
     `EXPECTED_TOOLS` in `tests/test_invariants.py`.
   - Tables persisted: `_SCHEMA` in `store.py`.
   - Make targets: `Makefile`.
   - Settings and defaults: `config.py`, `.env.example`.
   - Pins: `manifest.toml`, `docker/*.yml`, `docker/gateway.Dockerfile`.
3. For each mismatch decide which side is wrong. The PRD outranks
   architecture, and code outranks every other doc *unless* the code
   contradicts the PRD. That case is a bug report to the user, not a doc edit.
4. Fix docs only. Do not edit `src/` or `tests/`. Do not rewrite
   `docs/decisions/` files; they are a historical record.
5. Report each mismatch as one line: file, claim, what the code does, and
   what you changed or why you did not.
