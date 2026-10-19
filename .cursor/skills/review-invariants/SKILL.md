---
name: review-invariants
description: Reviews an AssistAI diff only for violations of the security model (broker bypass, taint laundering, unstaged writes, cross-person leaks, scope creep, secrets). Use when the user asks for a security or invariant review of a branch, PR, or uncommitted changes.
disable-model-invocation: true
---

# Review invariants

You are a reviewer, not the author. Do not fix anything; report findings.
Run as a fresh agent or subagent so you have not seen the author's reasoning.

1. Get the diff: `git diff main...HEAD` (or `git diff` for uncommitted work).
2. Run `uv run pytest -q tests/test_invariants.py`. A failure is a finding.
3. Read every changed file in `src/` in full, not only the hunks.
4. Check each question. Cite file and line for each finding.

## Questions

- **Broker bypass.** Can any path run a tool handler without
  `BoundSurface.execute` or `commit`? Look for direct handler calls and new
  registries.
- **Taint laundering.** Can untrusted text reach model context without
  `untrusted=True`? Can a path clear or skip the label (summaries, job
  reports, relay records, nested agent results)?
- **Unstaged writes.** Does anything change mail, calendar, lists, jobs, or
  the other phone without a stored proposal and a yes?
- **Report-only.** Can a scheduled job reach a sink, a staging tool, or
  `job_*`, directly or through something it calls?
- **Cross-person.** Can one assistant read the other's history, mail, private
  lists, or future per-person data? Is "not found" indistinguishable from
  "not yours"?
- **Caller identity.** Is the acting agent taken from `using_agent` or the
  surface, never from model-supplied arguments?
- **Scope creep.** Any new Gmail scope, provider permission, published port,
  or network path from a sidecar to the state volume?
- **Secrets.** Any key, token, or password in code, logs, test fixtures, or a
  prompt? Settings use `SecretStr`.
- **Dependencies.** Any new runtime dependency or unpinned image or action?

## Report

```
BLOCKING — must fix before merge
- path:line — what can happen, and to whom

CONCERN — worth a decision record or a test
- path:line — …

CLEAN — questions with no findings
```
