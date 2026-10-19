---
name: add-tool
description: Adds a brokered tool to AssistAI end to end, covering spec, catalog classification, handler binding, staging validation and preview, ACL, tests, and docs. Use when adding, renaming, or reclassifying any tool the model can call.
---

# Add a tool

Copy this checklist and keep it updated:

```
- [ ] Spec names the tool and its classification
- [ ] ToolSpec + parser in the domain package
- [ ] Catalog entry in builtin_catalog()
- [ ] Handler bound in gateway._start_signal
- [ ] Staging: _staging_error branch + staging.py preview
- [ ] EXPECTED_TOOLS updated in tests/test_invariants.py
- [ ] Behaviour tests in tests/test_<domain>.py
- [ ] config/assistai.example.toml and system_prompt_for, if agents should use it
- [ ] docs/architecture.md permissions table and section
- [ ] make check green
```

## Classify first

Answer these before writing code, and put the answers in the spec:

- **trusted** — Could the result contain text from outside the household
  (web, mail, invites, list items, a relay)? Then `trusted=False`.
- **sink** — Does it change anything outside this conversation? Give it a
  sink name `area:verb` (`gmail:file`, `calendar:write`).
- **staging** — Every sink stages. `test_every_sink_is_staged` fails otherwise.
- **web** — Does it reach the public internet? Then `web=True`, and agents
  need `web_access = true` as well as the tool on their list.

A tool may not be both untrusted and a sink. Split it into a read and a write.

## Wire it

- Spec and argument parser live in the domain package
  (`src/assistai/<domain>/tools.py`). Parsers raise the domain's
  `AssistAIError` subclass with a message the model can relay.
- Register with `_unbound` in `builtin_catalog()` and bind the live handler in
  the gateway, the way `bind_mail` does. The domain package must not import
  `broker`, `signal`, `store`, `gateway`, or `scheduler`.
- Staging tools: add a branch to `_staging_error` that runs the same parser
  the handler uses, so a bad call is refused before it is proposed. Add a
  readable preview in `staging.py` if the generic one would show raw JSON.
- Jobs: staging tools and sinks are hidden from report-only surfaces
  automatically. Do not special-case them.

## Test it

At minimum: the ACL denies it to an agent without it, the handler does not
run when denied, an untrusted result taints the next turn, and a staging call
does not run until commit. Copy the shapes in `tests/test_broker.py`.
