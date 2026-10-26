# AGENTS.md

Household assistant over Signal: two people, one bot number, a Raspberry Pi 5.
Python 3.12, asyncio, SQLite, Fireworks for inference. This file is a map.
The documents it points to are the source of truth.

## Read first

| Question | Where |
| --- | --- |
| What must the product do? | [docs/prd.md](docs/prd.md). Wins over everything else. |
| How is it built, and why? | [docs/architecture.md](docs/architecture.md) |
| What was decided after the PRD? | [docs/decisions/](docs/decisions/README.md) |
| What is being built now? | [docs/specs/](docs/specs/README.md) |
| How does it reach the Pi? | [docs/pi-deploy.md](docs/pi-deploy.md) |
| What shipped? | [docs/release-notes.md](docs/release-notes.md) |

If code and docs disagree, stop and say so. Do not silently pick one.

## Commands

```bash
make setup     # uv sync
make check     # ruff format --check, ruff check, mypy --strict, pytest (no network)
make fmt       # apply formatting and safe lint fixes
make live      # optional tests against real Fireworks (costs tokens)
make eval      # golden behaviour cases against real Fireworks (costs tokens)
make compare   # score candidate models against the real tool schemas
```

`make check` must pass before you call work done. Run the narrowest test
(`uv run pytest -q tests/test_broker.py`) while iterating, then the whole thing.

## Layout

```
src/assistai/inference/   model client and tool loop; knows nothing about tools' meaning
src/assistai/broker.py    the security boundary: ACL, taint, staging, audit
src/assistai/staging.py   proposal previews and yes/no parsing
src/assistai/signal/      transport, sender policy, the per-turn channel
src/assistai/{calendar,mail,research,shared}/   provider domains
src/assistai/store.py     the one SQLite file; schema version and migrations
tests/                    fakes for every external service; no network by default
evals/                    golden cases run by `make eval`
```

## Non-negotiables

These are enforced by `tests/test_invariants.py` where a test can enforce
them. The rest are on you.

- **Agents act only through the broker.** A new capability is a brokered tool
  with explicit `trusted`, `sink`, `web`, and `staging`. Never call a handler
  directly from the loop or the channel.
- **Deny by default.** New tools and grants are opt-in per agent in
  `config/assistai.toml`. `web_access` defaults to false and stays that way.
- **Anything that writes outside the conversation stages.** The person sees
  the stored call and says yes. Jobs never stage; they report.
- **Untrusted is sticky.** Web, mail, calendar text, list text, and relays are
  untrusted. That label persists with the message. Do not add a path that
  clears it early.
- **No send, no delete.** Gmail stays on `gmail.modify`. Never request a
  broader scope.
- **No new runtime dependency without a reason** in the commit message. Pin
  everything external in `manifest.toml` or `uv.lock`.
- **Secrets never enter the repo, logs, or model context.** Settings use
  `SecretStr`. The calendar password lives in the Keychain or a mode-600 file.
- **Schema changes bump `SCHEMA_VERSION`** in `store.py` with a migration and
  a test that opens the previous version.

## Working style

- Spec before code for anything bigger than a bug fix. Copy
  [docs/specs/TEMPLATE.md](docs/specs/TEMPLATE.md). No acceptance criteria,
  no implementation.
- Ask, or push back, when the request conflicts with the PRD or a decision.
- Small steps. One logical change per commit.
- Tests use the fakes in `tests/` (`fakes.py`, `agent_fakes.py`,
  `signal_fakes.py`, `channel_fakes.py`). A test that needs the network is
  marked `live`.
- Match the surrounding code: frozen dataclasses, `Protocol` seams, structlog
  events named `area.event`, docstrings that state the constraint, not the
  mechanics.
- Comments say why a constraint exists. They do not narrate the next line.
- When behaviour changes, update the doc that describes it in the same commit.

## Review

- The author never reviews its own change. Before a PR leaves draft, run
  `review-pr`: fresh subagents review the spec and the diff, a verifier
  tries to refute each finding, and every survivor gets a disposition in
  the PR body.
- Bugbot reviews the PR on GitHub once it is ready, using
  [.cursor/BUGBOT.md](.cursor/BUGBOT.md). `.cursor/rules` do not reach it;
  a rule Bugbot must enforce goes in that file too.
- No open blocking finding at hand-off. The person merges; agents do not
  approve or merge their own PRs.

## Commits

One sentence, present tense, saying what now holds and why it matters:

```
Stage calendar adds and relay drafts so a yes sends the stored text.
```

No prefixes, no ticket numbers. A dependency addition says why in the body.

## Skills in this repo

`.cursor/skills/` holds the repeatable workflows: `write-spec`, `add-tool`,
`review-pr`, `review-invariants`, and `doc-garden`. Use them by name.
