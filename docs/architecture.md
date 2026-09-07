# AssistAI Architecture

A self-hosted, multi-agent assistant reachable over Signal, running open models via
Fireworks AI, built to move from a development Mac to a Raspberry Pi 5 appliance.

## Design decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Language | Python 3.12 | Maintainer fluency on security-critical code; strongest HTML extraction ecosystem |
| Transport | Signal via `signal-cli` | Only messaging channel required |
| Inference | Fireworks AI | OpenAI-compatible, open-weight models |
| Search | Self-hosted SearXNG | No third party sees household queries; no API key |
| Headless browser | Deferred | Fetch + extract covers most research at a fraction of the cost |
| Target host | Raspberry Pi 5, 8 GB | Gateway, signal-cli, and SearXNG concurrently, with headroom |

## Security boundary

**The tool broker is the security boundary, not the agent process.**

Agents cannot execute code. They act only through brokered tools with explicit
per-agent allowlists. Every grant is opt-in, including `web_access`: an agent
added to the roster without a stated position on the internet does not get it.
This makes the agent loop an LLM conversation with no
independent authority, so per-agent containers would add Pi overhead without
buying isolation. Instead, the components that touch untrusted input or
untrusted code get their own containers and their own network.

If code execution is ever added, that is the point at which per-agent runner
containers become necessary. Revisit this decision then, not before.

## Topology

```
   your phone            her phone
        \                   /
         \                 /
      Signal (dedicated bot number)
                 |
        +--------v---------+
        |  signal-cli      |  container, MODE=json-rpc
        +--------+---------+
                 | REST/WS, private docker network
        +--------v------------------------------+
        |  GATEWAY  (Python, loopback only)     |
        |                                       |
        |   router  -->  sender allowlist       |
        |                + pairing              |
        |      |                                |
        |   +--v---+  +--------+  +----------+  |
        |   |jacob |  | spouse |  |organizer |  |  agent workers
        |   +--+---+  +---+----+  +----+-----+  |
        |      +----------+------------+        |
        |            +----v-----+               |
        |            |  BROKER  |  ACL + taint  |
        |            +----+-----+               |
        +-----------------+---------------------+
                          |
        +---------+-------+----+--------------+
        |         |            |              |
   +----v---+ +---v----+  +----v-----+  +-----v------+
   | stores | |searxng |  |  fetch   |  |  renderer  |
   | SQLite | |        |  | +extract |  | (deferred) |
   +--------+ +--------+  +----------+  +------------+
                          +--- egress-restricted net --+
                          |
                    Fireworks API
```

## Agents and permissions

| Agent | Binds to | Reads | Writes | Web access |
| --- | --- | --- | --- | --- |
| `jacob` | your Signal DM | own store, shared | own store, publish to shared | yes |
| `spouse` | her Signal DM | own store, shared | own store, publish to shared | yes |
| `organizer` | group chat, scheduled | shared only | shared | **no** |

The asymmetry is deliberate. The agent holding the most write authority over
household data never ingests untrusted content, and the agents that browse can
only *propose* to shared state through an explicit publish call. This removes
the highest-value injection path structurally rather than defensively.

Shared state is a SQLite database with typed tables and enforced ACLs, not a
directory of files. A workspace directory is a default working directory, not a
sandbox: absolute paths escape it.

## Model selection

Pinned in `manifest.toml`. Tool-calling reliability is the only capability axis
that matters much here: every action an agent takes is a brokered tool call over
a streamed response, so a model that emits malformed or non-standard tool-call
deltas breaks the architecture regardless of how well it writes prose.

MiniMax M3 is primary. DeepSeek V4 Flash runs the quarantined summarizer, chosen
for native structured output and for being a different model lineage than the
primary, so one injection technique is less likely to defeat both layers.

Swapping the model must stay a one-line manifest change. Phase 3 runs the
candidates in `models.candidates` against the real tool schemas, because
benchmarks do not predict how a model handles a specific toolset.

## Research tools

Three tiers, in increasing order of cost and risk.

1. **`web_search`** — SearXNG container. Returns titles, URLs, snippets only.
2. **`web_fetch`** — HTTP GET plus readability extraction to markdown. SSRF-hardened:
   private and link-local ranges rejected, redirects re-validated at each hop,
   hard caps on bytes and elapsed time.
3. **`browser_render`** — deferred. Headless Chromium in an isolated container,
   read-only, no cookie jar and no credentials, only for pages that require JS.

## Prompt injection defenses

Browsing admits attacker-controlled text into model context. Four layers:

**Taint tracking with blocked sinks.** Every tool result carries a trusted or
untrusted label, and the label is stored on the message. Taint is therefore
scoped to the conversation, not the turn: attacker text stays in history after
the turn that fetched it, so "fetch a poisoned page now, ask for a write in the
next innocent-looking message" has to fail too. Assistant replies written with
untrusted content in context inherit the label, because a summary carries an
injection as well as the page does. Taint clears only when every labelled
message has fallen out of the trimmed window. While tainted, the broker refuses
privileged sinks (writes to shared, messages to the other person, any state
mutation) *before* the tool executes. The agent can still read, summarize,
and answer.

**Quarantined summarizer.** Heavy pages are read by a separate cheap Fireworks
call that emits a structured result against a fixed schema. The primary agent
sees only the structured object, never the raw page.

**Nonce-delimited boundaries.** Page-sourced text is wrapped in markers carrying
a per-process random nonce, so a page cannot forge the delimiter. Untrusted
content is placed mid-prompt: models attend most strongly to the beginning and
end of context, which is where injected instructions do the most damage.

**Network isolation.** Fetch and render containers sit on a network with no
route to the gateway, the stores, or the Fireworks credential.

## Update awareness

`manifest.toml` pins every external dependency by version and digest. A watcher
diffs upstream against the manifest on a schedule and sends a summary over
Signal to the operator only. Nothing is applied automatically. `signal-cli` is
flagged at higher priority because upstream warns that old releases break as
Signal's server APIs change.

## Explicitly out of scope

No web UI, no plugin system or marketplace, no shell tool, no cron-as-a-tool, no
mobile nodes, no channels beyond Signal. Each of these is a documented source of
blast radius in comparable projects.

## Build phases

| Phase | Deliverable | Done when |
| --- | --- | --- |
| 0 | Skeleton | `docker compose up` yields a gateway logging a heartbeat |
| 1 | Fireworks loop | Multi-turn conversation from the terminal (`make chat`) |
| 2 | Signal | Texting the bot number gets a reply |
| 3 | Agents + broker | Two numbers reach two agents with distinct, empty tool sets; candidate models compared on real schemas |
| 4 | Shared store | Organizer reads both published feeds; neither personal store leaks |
| 5 | Research tiers 1-2 | Cited answer, with a write refused under taint in the audit log |
| 6 | Update watcher | Correctly flags a stale signal-cli and takes no action |
| 7 | Pi migration | Survives reboot and a week unattended |
| 8 | Household tools | Calendar, lists, reminders, each ACL'd per agent |
