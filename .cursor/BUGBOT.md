# AssistAI review rules

A household assistant over Signal for two people. The model is untrusted;
`src/assistai/broker.py` is the security boundary. `AGENTS.md` and
`docs/architecture.md` describe the model; `docs/prd.md` wins over both.

## Blocking

Flag these as bugs, with the path that reaches them:

- A tool handler runs without `BoundSurface.execute` or `commit`, or a
  handler is called directly from the inference loop, the Signal channel, or
  a job.
- A tool in `builtin_catalog()` without an entry in `EXPECTED_TOOLS` in
  `tests/test_invariants.py`, or a `ToolMeta` with `sink` set and `staging`
  false.
- Text from the web, mail, calendar, shared lists, relays, or job output
  reaching model context without `untrusted=True`, or any path that clears
  a conversation's taint before its history does.
- A write to mail, calendar, lists, jobs, or the other person's phone that
  runs without a stored proposal and a yes. A scheduled job that can reach
  a sink, a staging tool, or `job_*`.
- The acting agent taken from model-supplied arguments instead of
  `using_agent` or the bound surface.
- One person's history, mail, or private lists readable by the other, or a
  "not found" that differs from "not yours".
- A Gmail scope other than `gmail.modify`, a new published port, or a
  sidecar that can reach the state volume.
- A secret in code, logs, fixtures, or a prompt; a setting holding one that
  is not `SecretStr`.
- A schema change without a `SCHEMA_VERSION` bump, a migration, and a test
  that opens the previous version.
- `web_access` defaulting to true, or a tool or grant enabled for an agent
  that did not opt in through `config/assistai.toml`.

## Worth a comment

- A new runtime dependency, or an unpinned image, action, or model.
- Blocking calls inside async code; a task created without being awaited or
  tracked; a SQLite write outside a transaction.
- A broad `except` that swallows an error the person should hear about.
- Behaviour changed with no update to the doc that describes it.
- A test that would still pass if the handler under test never ran.

## Ignore

Formatting, import order, and type annotations: `make check` runs ruff and
`mypy --strict` in CI.
