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

Phase 3 of 8. Two Signal numbers reach two agents with distinct identities and
empty tool allowlists. The broker is the security boundary: a tool the model
invents is refused before any handler runs. Pairing still admits a number; it
does not assign an agent.

| Phase | Deliverable | State |
| --- | --- | --- |
| 0 | Skeleton: packaging, container, logging, tests | done |
| 1 | Fireworks inference loop | done |
| 2 | Signal channel | done |
| 3 | Three agents and the tool broker | done |
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
make up                   # gateway + signal-cli, no published ports
make logs                 # ctrl-c to detach; `make down` to stop
```

### Signal setup

Use a dedicated bot number, never a personal one. `signal-cli` ignores messages
from its own account, so a personal number cannot text its own assistant.

1. Start the stack (`make up` or `make up-dev`).
2. Link the container as a device of that number, from inside the compose
   network so 8080 stays off the LAN:

   ```bash
   make signal-link
   ```

   Scan the printed URI from Signal → Settings → Linked devices.
   To register a new number instead: `python -m assistai signal register +1…`
   (usually needs a captcha from [signalcaptchas.org](https://signalcaptchas.org/registration/generate.html)).
3. Set in `.env`:

   ```
   ASSISTAI_SIGNAL_ACCOUNT=+15555550100
   ASSISTAI_SIGNAL_ALLOW_FROM=+15555550101
   ```

4. Copy the agent roster and put real numbers in:

   ```
   cp config/assistai.example.toml config/assistai.toml
   ```

   Each Signal DM binding is a different agent. Pairing admits a number; it does
   not give them someone else's assistant, and an allowed number with no binding
   reaches no agent at all. Every permission is opt-in, `web_access` included.
   Restart the gateway. Text the bot from a bound phone.

Unknown numbers receive a pairing code. Approve from an **operator** phone,
meaning one listed in `ASSISTAI_SIGNAL_ALLOW_FROM`:

```
/approve 482193
```

A number admitted this way can talk to the bot but cannot approve anyone else,
so letting one person in never hands out the ability to let others in.

Per-sender limits cap what a single number can spend: 12 turns per minute and
4000 characters per message by default. Strangers get at most 3 pairing replies
per hour, so a flood cannot turn the bot number into an outbound spam source.

`make compare` scores the primary and every candidate in `manifest.toml` against
the real `get_time` schema. That is the check that matters for swapping models;
published benchmarks do not predict our JSON.

`make up-dev` binds signal-cli to `127.0.0.1:8080` so host-mode `make run` can
reach it. Do not use that overlay on the Pi.

## Layout

```
docs/architecture.md      design, threat model, phase plan
manifest.toml             pinned external dependencies; the update watcher reads this
config/                   agent roster; copy assistai.example.toml to assistai.toml
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
