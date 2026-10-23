---
name: write-spec
description: Drafts a build-phase spec in docs/specs/ from a feature request, with checkable acceptance criteria and a security review. Use before implementing any AssistAI feature larger than a bug fix, or when the user asks to plan, scope, or spec a phase.
disable-model-invocation: true
---

# Write a spec

Planning only. Do not edit `src/` or `tests/` while running this skill.

1. Read `docs/prd.md`, `docs/architecture.md`, and every file in
   `docs/decisions/`. Find the PRD section this work implements. If there is
   none, stop and tell the user the PRD needs an amendment first.
2. Read the code the change touches. Name real modules, functions, and tables
   in the Design section, not invented ones.
3. Copy `docs/specs/TEMPLATE.md` to `docs/specs/NNN-short-name.md`, with NNN
   one past the highest existing spec or architecture phase.
4. Fill every section. Acceptance criteria must each name the test, eval
   case, or manual step that proves them. "Works well" is not a criterion.
5. Security review: for every new or reclassified tool, state `trusted`,
   `sink`, `web`, `staging`, and what happens under taint and in a
   report-only job. State what the other person's assistant can learn.
6. List open questions. Use the AskQuestion tool for the ones only the user
   can answer. Leave status `draft` until they are closed.
7. Push back. If the request conflicts with a PRD principle or makes the
   security model harder to reason about, say so in the spec and in chat.

Finish by reporting the spec path and the open questions, nothing else.
