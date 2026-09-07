# AssistAI

A self-hosted, multi-agent assistant reachable over Signal, running open models
through Fireworks AI. Developed on a Mac, deployed to a Raspberry Pi 5 as an
always-on appliance.

Three agents share one gateway: one for each of us, plus an organizer for shared
household state. They have different read, write, and tool permissions, enforced
by a broker rather than by convention.

See [docs/architecture.md](docs/architecture.md) for the design and the
reasoning behind it.

## Status

Phase 1 of 8. The gateway validates pinned Fireworks models at startup. A
terminal REPL holds a multi-turn conversation, including a side-effect-free
`get_time` tool so the tool-call loop can be exercised.

| Phase | Deliverable | State |
| --- | --- | --- |
| 0 | Skeleton: packaging, container, logging, tests | done |
| 1 | Fireworks inference loop | done |
| 2 | Signal channel | |
| 3 | Three agents and the tool broker | |
| 4 | Shared store with ACLs | |
| 5 | Research tools: search, fetch, extract | |
| 6 | Update watcher | |
| 7 | Raspberry Pi migration | |
| 8 | Household tools | |

## Prerequisites

- Python 3.12 (`uv` installs it for you)
- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- Docker Desktop, for the container workflow

## Quick start

```bash
cp .env.example .env      # set FIREWORKS_API_KEY
make setup
make check                # lint, types, tests (no network)
make chat                 # multi-turn conversation
make run                  # daemon: validate models, then heartbeat
```

Containerized:

```bash
make lock                 # required before the first image build
make build
make up
make logs                 # ctrl-c to detach; `make down` to stop
```

## Layout

```
docs/architecture.md      design, threat model, phase plan
manifest.toml             pinned external dependencies; the update watcher reads this
config/                   agent definitions and tool ACLs (example is a design artifact)
docker/                   Dockerfile and compose
src/assistai/             gateway source
tests/                    test suite
```

## Conventions

**Dependencies are attack surface.** The runtime dependency list is short on
purpose. Adding to it needs a reason in the pull request.

**Pin everything external.** Images, models, and the Python lockfile are pinned
in `manifest.toml` and `uv.lock`. The update watcher notifies; it never applies.

**Secrets live in `.env`,** which is gitignored, and are read as `SecretStr` so
they do not leak into logs or tracebacks.

**No published ports.** Nothing in the stack listens on the LAN. Operator access
is over SSH or a tailnet to loopback.
