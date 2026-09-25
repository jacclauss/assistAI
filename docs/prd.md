# AssistAI Product Requirements

Household assistant over Signal. Two people, one bot number, a box at home.
This document is the product source of truth. [architecture.md](architecture.md)
is how we build it; when they disagree, this file wins and the architecture
must be updated.

## Problem

ChatGPT on a phone cannot see the shared calendar, cannot file mail, and cannot
put a lookup on the other person's phone without copy-paste. A household
assistant that texts you has to do those jobs without becoming a stranger's
chatbot, without sending or deleting mail, and without either person talking to
the other's assistant.

## Who it is for

| Person | Role |
| --- | --- |
| Jacob | Operator. Texts his assistant. May admit config, credentials, and jobs. |
| Spouse | Equal user of the same jobs. Texts her assistant. Does not need SSH, pairing, or config. |

Guests are out of scope. Unknown Signal numbers hear nothing. There is no open
channel from one person to the other person's assistant.

## Jobs

On a normal day, either person texts the bot:

1. **Anything important in email?** Look now, or at a time they scheduled.
2. **What's on the docket today?** The shared household calendar, first.

They also need to **relay** something the calendar app will not already show:
a lookup summary, a document, a note. That is a message to the other person's
phone, not a conversation with the other assistant.

One Signal message may contain more than one ask. The bot must attempt all of
them, not drop the rest after the first tool call.

## Principles

**Deny by default.** Mail is not sent or deleted. Unknown numbers get silence.
Cross-person traffic is a relay, never a session.

**Irreversible and cross-boundary actions stage first.** Proposed trash goes to
a folder before delete is ever granted. A relay shows a draft, waits for
yes, then sends those bytes. A critique replaces the draft; it does not send.

**Scheduled work talks. Background work dies.** An indefinite schedule (daily
email) always produces a Signal message. A silent background watch (flights)
must be requested explicitly and must have a max lifetime so nothing stale
runs forever. Failures never fail closed with silence: Gmail down, calendar
down, a job that cannot run — ping the owner.

**Her assistant is hers.** A relay does not go *through* her model to reach her.
It does go *into* her context after it is delivered, so she can ask follow-ups
without pasting.

## In scope

- Signal DMs to a dedicated bot number, allowlist only.
- Two agents (Jacob, spouse), each bound to one number.
- Email, starting with Jacob's inbox; hers as soon as credentials exist. Read,
  summarize, archive, star, move, draft. Staged-trash folder as a move target.
  No send. No delete.
- Shared household calendar. Reads are immediate. Adding an event stages
  and waits for a yes. More calendars later.
- First-class jobs: create, list, change schedule, cancel, by talking to the
  bot ("stop the morning email check").
- Research: "look this up" via search + fetch, cited, in Signal. In MVP.
  Headless Chromium is not.
- Relay: confirm → verbatim Signal send → record in the recipient's history.
- Update awareness: flag stale pins, apply nothing.
- Raspberry Pi as the always-on host.

## Out of scope

No web UI, plugins, shell, extra chat networks, cameras, voice, reading phone
notifications, or acting as either person on the public web (booking, forms).
No open session with the other assistant. No pairing by default. Headless
browser stays deferred. Email send and email delete stay deferred until staged
trash has been lived with.

## Requirements

### Signal

- Only `ASSISTAI_SIGNAL_ALLOW_FROM` (and later explicit bindings) get a reply.
- Unknown senders: no outbound message.
- Each bound number reaches only its own agent.
- One inbound text may fan out to several tools and jobs.
- Relays may carry a Signal attachment (document, image) or a link, not
  text-only. The current receive path that ignores attachments is wrong for
  this product: inbound attachments must be accepted when the user is relaying,
  and outbound relays must send them. The recipient's assistant must know that
  a relay happened and what it was (body + attachment identity / extracted
  text), so neither person has to leave the bot thread to share context.

### Email

- Provider is **Gmail** (Google). Not iCloud Mail.
- On demand when asked; also as a scheduled job.
- "Important" may be model-judged against the inbox the user named; the user
  can later tighten this with sender rules.
- Allowed: read, archive, star, move (including a human-reviewed trash folder),
  draft **into Gmail** (a real draft the user can open and send themselves).
- Forbidden: send, permanent delete.
- Mail bodies are untrusted in the conversation, same as a fetched page.
- **Filing stages as a batch.** Reading mail makes the conversation untrusted,
  and filing is a mutation, so the agent proposes the list ("archive these 6
  newsletters?") and one confirmation covers the batch. Not one prompt per
  message, and not a silent bulk move. The user sees what will be touched.
- Start with one inbox if both credentials are not ready; the product is both.

### Calendar

- Provider is **Apple / iCloud** (CalDAV), the shared household calendar.
- Do not require migrating calendars onto Google. Dual login (Google mail +
  Apple calendar) is the intended setup; collapsing to one vendor is a fallback
  if Apple auth on a headless Pi is unworkable, not the plan.
- v1 is that shared calendar, not a merge of personal calendars.
- "What's on the docket today" is a read of that calendar.
- Adding an event stages. The person replies yes, and the stored event is
  what gets written. The model cannot pick another calendar or add attendees.
  Jobs still only read.
- Calendar writes in the app already notify the other person; the bot does not
  need to announce those. Relays are for things the calendar will not show.

### Jobs

Two kinds, both brokered, both inspectable in conversation:

| Kind | Chatter | Lifetime |
| --- | --- | --- |
| Scheduled, user-facing (daily email) | Always a Signal message to the owner | Until cancelled |
| Background watch (e.g. flights) | Silent unless it fails or expires | Required max time |

- "What is running?" and "stop the morning email check" must work in Signal.
- A job failure (auth, network, tool error) messages the owner. No silent skip.
- An empty successful digest still messages, because the schedule is
  user-facing. A background watch that finds nothing stays quiet.
- **Jobs report; they do not act.** A job may read mail, calendar, and the web,
  and it messages the owner with what it found. It never files, drafts, writes,
  or relays, because nobody is present at 7am to approve a staged action. If
  the digest suggests filing, the user stages that from their reply.
- Jobs survive a restart. A schedule set last month still fires after a reboot.

### Memory

- Conversation history persists across restarts. A reboot must not make either
  assistant forget a relay, a digest, or what was decided yesterday.
- The untrusted label on a message persists with it. A restart must not clear
  taint, which in-memory history would do silently.
- History is a window per person, not forever. Taint clears when the labelled
  messages age out of that window.

### Relay

Not `message:other_peer` as a free tool, and not a pass through the recipient's
agent loop.

1. Sender's assistant drafts the Signal body (and any attachment). The sender
   can critique that draft; each revision is a new proposal.
2. Sender confirms the draft on screen.
3. Gateway sends those bytes and attachments to the recipient's number, labeled
   so it is obviously a relay (`From Jacob:`, not her assistant speaking).
4. Recipient reads it on the phone immediately (no extra LLM rewrite).
5. Gateway appends a structured record to the recipient agent's history:
   who sent it, the body, and enough about the attachment (type, name,
   extracted text or a short description) that her assistant can talk about it.
   Relaying must not require leaving the bot conversation.
6. Reply-in-thread without "tell Jacob …" is a normal turn with **her**
   assistant. To send something back she uses the same relay.

Taint: relay bodies are untrusted in the recipient's conversation (they may
carry a page, a mail, or the other person's words). Confirmation authorizes
delivery to the phone; it does not bless the content as a privileged sink on
her side. Her assistant still cannot write shared state, file her mail, or
relay onward until that taint has left the window — unless we later add an
explicit confirm for those sinks too.

### Agents

Jacob and spouse are the product. The organizer is a process that holds household
lists. It is not a person either of them texts, and it is not on the relay path.
A shared list is visible to both. A private list is visible only to the person
who created it; the other assistant gets the same answer as for a list that
does not exist. Publishing a private list is a separate staged change and waits
for a yes. A job may read the lists visible to its owner and cannot change them.

## Success

The product is working when, from a phone, with the Pi on:

- Either person can ask about today's shared calendar and get a faithful agenda.
- Either person can ask about email and get a summary plus archive/star/move/
  a Gmail draft, never a send or delete.
- A daily email job fires, texts the owner, and can be cancelled in Signal.
- If Gmail, iCloud calendar, or the job runner is down, the owner gets an
  error text, not silence.
- "Tell her this" (text, link, or attachment) shows a draft, and on yes she
  receives it and her assistant's next turn knows it was from him and what
  it was.
- A stranger texting the bot number gets nothing.
- "Look this up" returns a cited answer in Signal, without a headless browser.
- A lookup cannot text her, file mail, or write calendar without the
  corresponding confirm / allowlist.
- "Archive those newsletters" shows the list once and files them on one yes.
- The Pi reboots and both assistants still know yesterday's relay and digest.

## Open engineering (does not block starting)

Product decisions above are closed. These are implementation choices to make
in the phase that touches them, not more product interviews:

- Google OAuth vs a restricted app password; where refresh tokens live on the Pi.
- iCloud CalDAV with an app-specific password vs any other Apple path.
- Attachment size and MIME allowlist; how a PDF/image is summarized into
  history so the model can talk about it without swallowing the binary.
- Confirm UX when the next message is not a yes/no; how long a proposal stays
  open before it expires.
- History window size, and whether the store is encrypted at rest.
- Timezone for "every morning at 7," and whether a 3am failure waits for
  morning. Default is send immediately.
- "Watch flights" is an example of a TTL'd background job, not a flight-search
  product.

## Architecture implications

These contradict the current plan and must be reflected in `architecture.md`:

- **Jobs are in scope.** "No cron-as-a-tool" meant "the model does not get a
  raw cron primitive." A brokered job object with notify/TTL rules is required.
- **Relay replaces free cross-peer messaging.** Confirm + verbatim send +
  history inject. Recipient loop does not rewrite.
- **Research is MVP.** Search + fetch, cited, in Signal. Order relative to
  mail and calendar is a build convenience, not a scope cut.
- **Organizer is not the UX.** Two Signal identities, not three chats.
- **Dual providers.** Gmail + iCloud CalDAV. Do not collapse the household onto
  Google unless Apple auth on the Pi fails.
- **Attachments are in scope for relay.** `ignore_attachments` cannot stay.
