# AssistAI Architecture

A self-hosted, multi-agent assistant reachable over Signal, running open models via
Fireworks AI, built to move from a development Mac to a Raspberry Pi 5 appliance.

Product requirements live in [prd.md](prd.md). This file is how we build it.
If they disagree, the PRD wins and this file is wrong.

## Design decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Language | Python 3.12 | Maintainer fluency on security-critical code; strongest HTML extraction ecosystem |
| Transport | Signal via `signal-cli` | Only messaging channel required |
| Inference | Fireworks AI | OpenAI-compatible, open-weight models |
| Mail | Gmail API | The household's mail already lives there |
| Calendar | iCloud shared calendar over CalDAV | The household's calendar already lives there; not worth a migration |
| Durable state | One SQLite file in the state volume | History, taint, jobs, and staged actions must survive a reboot |
| Scheduling | In-process asyncio scheduler | One box, few jobs; a second daemon buys nothing |
| Search | Self-hosted SearXNG | No third party sees household queries; no API key |
| Headless browser | Deferred | Fetch + extract covers most research at a fraction of the cost |
| Target host | Raspberry Pi 5, 8 GB | Gateway, signal-cli, and SearXNG concurrently, with headroom |

## Security boundaries

There are two, and they answer different questions.

**The tool broker decides what an agent may attempt.** Agents cannot execute
code. They act only through brokered tools with explicit per-agent allowlists.
Every grant is opt-in, including `web_access`: an agent added to the roster
without a stated position on the internet does not get it. This makes the agent
loop an LLM conversation with no independent authority, so per-agent containers
would add Pi overhead without buying isolation. Instead, the components that
touch untrusted input or untrusted code get their own containers and network.

**Staging decides what actually happens.** Anything that mutates a mailbox or
crosses to the other person's phone is proposed, shown, and executed only on a
human yes. The broker can be fooled by a convincing model; a person reading
"archive these 6" cannot be fooled into approving a list they can see.

Narrow provider scopes are the third leg and cheaper than both: send and delete
are not requested from Google at all, so no bug in either layer can escalate
into a sent or destroyed mail.

If code execution is ever added, that is the point at which per-agent runner
containers become necessary. Revisit this decision then, not before.

## Topology

```
   your phone                her phone
        \                       /
         Signal (dedicated bot number)
                    |
          +---------v--------+
          |  signal-cli      |  container, MODE=json-rpc
          +---------+--------+
                    | REST/WS + attachments, private network
   +----------------v-------------------------------+
   |  GATEWAY  (Python, no published ports)         |
   |                                                |
   |   router --> sender allowlist                  |
   |        |                                       |
   |   +----v----+   +--------+                     |
   |   | jacob   |   | spouse |   agent loops       |
   |   +----+----+   +---+----+                     |
   |        +------------+                          |
   |               |                                |
   |        +------v------+  ACL + taint            |
   |        |   BROKER    |  + staged actions       |
   |        +------+------+                         |
   |               |                                |
   |   scheduler --+  report-only jobs              |
   +---------------+--------------------------------+
                   |
   +-------+-------+--------+---------+-----------+
   |       |                |         |           |
+--v---+ +-v-----+ +--------v+ +------v-+ +-------v--+
|state | | gmail | | icloud  | | searxng| | fetch    |
|SQLite| | API   | | CalDAV  | |        | | +extract |
+------+ +-------+ +---------+ +--------+ +----------+
                                 +-- egress-restricted --+
                                            |
                                      Fireworks API
```

## Agents and permissions

| Agent | Binds to | Reads | Writes | Web |
| --- | --- | --- | --- | --- |
| `jacob` | his Signal DM | own history, own Gmail, shared calendar | Gmail file/draft (staged), relay (staged) | yes |
| `spouse` | her Signal DM | own history, own Gmail, shared calendar | Gmail file/draft (staged), relay (staged) | yes |

Two agents are the product. Each is bound to exactly one number and reaches
nothing else.

An **organizer** is a process, not a Signal identity. It owns the household
lists and never browses, so the component with the broadest write authority
never ingests the open web. Nobody texts it, and relays do not pass through it.
The roster is two Signal DMs; a group binding, or an agent named organizer, is
rejected at load. Each list is either shared or private to one agent. The `own`
grant covers that agent's private lists. The `shared` grant covers lists both
can see. A named lookup that misses stays inside the scope that was asked for,
so one person cannot learn that the other has a private list of that name. A
list change stages and waits for a yes, including the change that publishes a
private list. Reading a list taints the turn, because item text can carry
instructions. A tainted turn can still propose a change; the preview says so,
and the stored call is what runs.

Shared state is a SQLite database with typed tables and enforced ACLs, not a
directory of files. A workspace directory is a default working directory, not a
sandbox: absolute paths escape it.

## Providers

| Need | Provider | Auth |
| --- | --- | --- |
| Mail | Gmail API | Google OAuth per person; refresh token in the state volume |
| Calendar | iCloud shared calendar | CalDAV with an app-specific password |

Two vendors is the intended setup, not an accident. Collapsing the household
onto Google is the fallback if Apple auth on a headless Pi proves unworkable,
not the plan.

Credentials are per person and selected by agent: jacob's tools use jacob's
token. The operator has filesystem access to both. That is an accepted property
of a single-box household appliance, not something the broker can fix.

The shared calendar is the exception: one Apple ID that can see that calendar,
one app-specific password, and the calendar's display name. The model does
not choose another calendar. A read lists one day. Adding an event stages
and waits for a yes; it does not invite anyone. "Today" is a day in `ASSISTAI_TIMEZONE`. iCloud is asked to expand
recurrences into that local day. Event text is untrusted, because an invite is
attacker-controlled text. The app-specific password is not stored in the repo
or in `.env`. The Mac gateway reads it from the Keychain; the Pi reads a
mode-600 file outside the repo. The model never receives it.

Requested Gmail scopes cover read, label, archive, star, move, and draft.
`send` and permanent `delete` are never requested, so the capability does not
exist in the process at all.

## Persistence

Everything that must outlive a reboot is in one SQLite file in the gateway
state volume:

- conversation history per agent, including the untrusted label on each message
- pending staged actions
- job definitions, schedules, TTLs, and last-run state
- relay records
- the sender allowlist

Two consequences worth stating plainly. A restart does not forget that a relay
happened, so her assistant can still answer "what did he send me yesterday?"
And a restart does not launder taint: in-memory history meant a reboot silently
cleared every untrusted label, which was a real hole once mail and the web are
in scope.

History is a window per agent, bounded by count and age. Taint clears when the
labelled messages fall out of that window — the same rule as before, now
durable rather than incidental.

The file holds mail summaries, relayed documents, and provider tokens. It is
not encrypted at rest; the Pi's disk is the trust boundary.

## Staged actions

Mail filing and relay are the same primitive.

A tool marked as staging never executes when the model calls it. The broker
records the **resolved call** — tool name and arguments, not the model's prose —
and returns a proposal for the user. The next inbound message either confirms
it or does not. On yes, the gateway executes the stored call, so what the user
approved is exactly what runs. The model does not get a second chance to
re-render the action between the preview and the execution.

Properties that matter:

- One proposal may cover many items, so "archive these 6" costs one
  confirmation rather than six.
- Proposals expire, and a newer proposal replaces an older one, so a stale yes
  cannot fire something the user has forgotten about.
- Under taint the proposal is labelled as having been suggested while untrusted
  content was in context. The human is the check, so they should know that a
  page or a mail is what asked for this.
- A scheduled job never stages and waits. Jobs report.

## Jobs

The model does not get raw cron or a shell. It gets brokered job objects,
stored in SQLite and inspectable in conversation.

| Kind | Chatter | Lifetime |
| --- | --- | --- |
| Schedule (daily digest) | Always messages the owner | Until cancelled |
| Watch (e.g. flights) | Silent unless it finds, fails, or expires | Required TTL |

Jobs are **report-only**. A job may read — mail, calendar, the web — and it
messages the owner with what it found. It never files, drafts, writes, or
relays, because nobody is present to approve a staged action at 7am. If a
digest suggests filing, the user stages that from their reply, where they can
see the list.

A job runs as one agent, with that agent's ACL and that person's credentials.
Failures (auth, network, tool error) message the owner; there is no silent
skip. An empty schedule run still messages, because the person asked for a
check at that time. An empty watch stays quiet. A watch without a TTL is
rejected at creation.

Job output is appended to the owner's conversation history, so "what was in the
digest?" works without re-fetching.

This makes the channel no longer strictly reply-to-inbound: the gateway may
send to a bound number with no inbound trigger. Those sends go only to bound
numbers and are metered separately from user turns, so a misbehaving job cannot
consume a person's conversational rate limit.

## Attachments

Relay carries documents and images eventually, but this phase does not pull
file bytes onto the Pi. Receive keeps `ignore_attachments=true` so the
websocket does not carry binaries; signal-cli still lists name, type, and
size on the envelope. Frames larger than `ASSISTAI_SIGNAL_MAX_RECEIVE_BYTES`
are dropped. Downloading attachment bytes, MIME-checking them, and extracting
text still wait; until then a file-only DM is turned into a prompt that asks
the sender to paste a link. Outbound relays this phase are text (and links
in that text). Web pages already go through the isolated extractor.

Text extraction runs in the same isolated extractor as web pages, so a
malformed PDF or image cannot exploit a parser inside the gateway. Mail HTML
goes through that same extractor for the same reason.

What persists is the extracted text plus the file's identity (name, type,
size), not the binary. Her assistant can discuss the document next week without
the gateway hoarding files.

## Model selection

Pinned in `manifest.toml`. Tool-calling reliability is the only capability axis
that matters much here: every action an agent takes is a brokered tool call over
a streamed response, so a model that emits malformed or non-standard tool-call
deltas breaks the architecture regardless of how well it writes prose.

MiniMax M3 is primary. DeepSeek V4 Flash runs the quarantined summarizer, chosen
for native structured output and for being a different model lineage than the
primary, so one injection technique is less likely to defeat both layers.

Swapping the model must stay a one-line manifest change. `make compare` runs the
candidates in `models.candidates` against the real tool schemas, because
benchmarks do not predict how a model handles a specific toolset.

## Research tools

Three tiers, in increasing order of cost and risk.

1. **`web_search`** — SearXNG container. Returns titles, URLs, snippets only.
2. **`web_fetch`** — HTTP GET plus readability extraction to markdown. SSRF-hardened:
   private, loopback, CGNAT, and link-local ranges rejected; redirects re-validated
   at each hop; the validated address is pinned and dialled directly so a second
   lookup cannot rebind to a private host; the body is streamed against a byte cap
   so a compressed bomb cannot expand into memory; and a whole fetch, redirects
   included, is bounded by one elapsed-time budget.
3. **`browser_render`** — deferred. Headless Chromium in an isolated container,
   read-only, no cookie jar and no credentials, only for pages that require JS.

## Prompt injection defenses

Browsing, mail, calendar invites, and relayed attachments all admit
attacker-controlled text into model context. Five layers:

**Staged actions.** The outermost layer, and the only one an attacker cannot
argue with. A poisoned mail that says "archive everything from the bank"
produces a visible proposal listing those messages, and the person says no.

**Taint tracking.** Every tool result carries a trusted or untrusted label, and
the label is stored on the message and persisted with it. Taint is scoped to
the conversation, not the turn: attacker text stays in history after the turn
that fetched it, so "fetch a poisoned page now, ask for a write in the next
innocent-looking message" has to fail too. Assistant replies written with
untrusted content in context inherit the label, because a summary carries an
injection as well as the page does. Taint clears only when every labelled
message has fallen out of the window. While tainted, the broker refuses any
mutation that is not going through staging, and marks the proposals that do.

**Quarantined summarizer.** Heavy pages are read by a separate cheap Fireworks
call that emits a structured result against a fixed schema. The primary agent
sees only the structured object, never the raw page.

**Nonce-delimited boundaries.** Page-sourced text is wrapped in markers carrying
a per-process random nonce, so a page cannot forge the delimiter. Untrusted
content is placed mid-prompt: models attend most strongly to the beginning and
end of context, which is where injected instructions do the most damage.

**Network isolation.** Fetch, extract, and render run in their own containers,
holding no Fireworks credential and no state volume, with the HTML parser kept
out of the gateway process entirely. They share a bridge with the gateway
because the gateway has to call them; the gateway listens on nothing, so the
reachability is one-directional in practice. The gateway refuses to start if
`web_fetch` is granted while no extract sidecar is configured, because that
would quietly move the parser back in-process.

### Relay specifically

Relay is not a free `message:other_peer` sink. The sender's assistant drafts
the message. A critique replaces that draft. Delivery is: confirm the draft
on screen → Signal sends those bytes → append a record to the recipient's
history so their assistant knows who sent it and what it said. Nothing is
rewritten between confirm and send. The injected record is untrusted on her side:
confirmation authorized the delivery to her phone, not her assistant's later
tool use.

## Update awareness

`manifest.toml` pins every external dependency by version and digest. A watcher
diffs upstream against the manifest on a schedule and sends a summary over
Signal to the operator only. Nothing is applied automatically. `signal-cli` is
flagged at higher priority because upstream warns that old releases break as
Signal's server APIs change.

## Explicitly out of scope

No web UI, no plugin system or marketplace, no shell tool, no mobile nodes, no
channels beyond Signal, no session with the other person's assistant, no email
send or delete, no acting as either person on the public web. Pairing is off by
default. Headless Chromium is deferred. Each of the omitted surfaces is a
documented blast radius in comparable projects.

## Deltas from the current build

Phase 3 shipped assumptions the PRD overturns. These are the concrete changes
that remain:

- Jobs can send without an inbound trigger, but they still cannot stage an
  action. A job may read mail, the shared calendar, and the web when it
  reports; it still cannot file, draft, write, or relay.

## Build phases

Persistence comes first because staging and jobs both need durable state, and
because taint that a reboot can clear is worse than no taint. Relay and jobs
land before the providers so the novel machinery is proven without waiting on
OAuth. Calendar precedes mail because CalDAV read-only is the smaller surface.
Calendar and mail can move earlier if credentials are ready sooner.

| Phase | Deliverable | Done when |
| --- | --- | --- |
| 0 | Skeleton | `docker compose up` yields a gateway logging a heartbeat |
| 1 | Fireworks loop | Multi-turn conversation from the terminal (`make chat`) |
| 2 | Signal | Texting the bot number gets a reply |
| 3 | Agents + broker | Two numbers reach two agents with distinct, empty tool sets; candidate models compared on real schemas |
| 4 | Persistence | History, untrusted labels, and the allowlist survive a restart; a reboot does not clear taint |
| 5 | Staged actions | A staging tool proposes, waits, and on yes executes the stored call, not a re-rendered one |
| 6 | Relay | A drafted message reaches the other phone after one confirm, and her assistant knows what arrived |
| 7 | Jobs | Schedule and TTL'd watch, report-only, cancellable in Signal, failures ping, survive a reboot |
| 8 | Research tiers 1-2 | Cited answer; a sink proposed under taint is labelled as such in the audit log |
| 9 | Calendar | "What's on the docket today" reads the iCloud shared calendar |
| 10 | Email | Digest, batch-staged filing, Gmail drafts; send and delete scopes never requested |
| 11 | Shared store | Shared and private lists with ACLs; organizer runs as a process and never browses |
| 12 | Update watcher | Correctly flags a stale signal-cli and takes no action |
| 13 | Pi migration | Survives reboot and a week unattended |
