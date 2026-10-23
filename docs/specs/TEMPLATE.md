# NNN — Title

Status: draft

Implements: [prd.md § Section](../prd.md)

## Outcome

One paragraph. What a person can do over Signal afterwards, or what holds
that did not hold before.

## Out of scope

What this phase deliberately does not do, so the implementer does not drift
into it.

## Design

The shape of the change: new modules, tools (with `trusted` / `sink` / `web` /
`staging`), tables, config keys. Link the architecture section it changes.

## Acceptance criteria

Each one is checkable. Name the test or eval case that proves it.

- [ ] AC1 — … (`tests/test_….py::test_…`)
- [ ] AC2 — … (`evals/cases/….toml`)

## Security review

- Which tools are new or reclassified, and why that classification.
- What happens under taint.
- What a job (report-only) can and cannot reach.
- What the other person's assistant can learn.

## Open questions

Close each before status moves to `ready`. A closed question that changes the
PRD or an invariant becomes a file in [decisions](../decisions/README.md).
