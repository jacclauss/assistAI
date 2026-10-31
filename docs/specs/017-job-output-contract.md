# 017 — A job-run contract and one checked path for model text

Status: ready

Implements: [prd.md § Jobs](../prd.md), [prd.md § Memory](../prd.md), and
[prd.md § Principles](../prd.md) ("Irreversible and cross-boundary actions
stage first"). Changes [architecture.md § Staged actions](../architecture.md),
[§ Jobs](../architecture.md), [§ Persistence](../architecture.md), and
[§ Prompt injection defenses](../architecture.md). Supersedes three points of
[spec 015](015-reminder-fixes.md), listed under Design.

Decision: [0003](../decisions/0003-reworded-reminders-wait-for-yes.md), a
reworded reminder waits for a yes. It extends
[0001](../decisions/0001-reminders-stage-when-tainted.md).

Depends on [spec 016](016-wall-clock-schedules.md), which ships first and owns
`SCHEMA_VERSION` 15 and the one formatter for local times.

## Outcome

Model text has a contract in both directions. Going out, every reply, job
report, and text placed beside a preview passes one function,
`check_outbound`, before it reaches Signal or history. That function refuses
a completion cut off at the token cap, empty text, tool or thinking markup,
oversize or self-repeating text, copies of the server's own preview and
summary wording, and claims of actions that did not run. Coming back, the
model never sees server text in its own voice. Previews, commit summaries,
job reports, and notices are stored as labelled records and shown to the
model as gateway records it cannot write. A job run no longer reads the
conversation. Code decides whether the run looked and what the tools found.
The run is shown its own earlier results, not `trusted_context`. A yes runs a
proposal only when it answers the preview the person was just shown.

Afterwards, for the household:

- The morning-email echo cannot recur. A report that hit the token cap,
  repeats itself, or carries `[job:` lines is never sent or stored. A report
  is never written from earlier reports either, because a run does not see
  them.
- A job that answers without calling a tool is a failed check, whatever words
  it uses. A watch whose tools found nothing stays quiet.
- A search that matched nothing is reported as nothing found, not as the
  whole inbox's unread total.
- "Yes" runs the proposal only if it is the next thing the person sends
  after that preview. Any other message first withdraws the proposal.
  "Yes!", "ok", and "sure" also count as yes. "cancel" sent in reply to a
  preview that cancels a job asks again instead of discarding it.
- The model cannot say "Done", "archived", or "cancelled" when nothing ran.
  It cannot write its own "Reply yes" either. The Oct 29 exchange, where the
  model typed out a cancel preview and then a "Ran 1 action" summary while
  the job stayed live, is refused at both steps.

## Out of scope

- Everything 016 owns: how a schedule is stored and its next run computed;
  time arguments of `job_create`, `relay`, and `job_reschedule`; the local
  time formatter itself; DST gaps and folds; the timezone at startup. This
  spec calls that formatter. It does not design it.
- Everything 018 owns: per-part Signal send status (C-9, D-8), proposal
  states instead of delete-then-run (E-2, X1-6), cancellation and shutdown as
  recorded outcomes (E-1, E-5), inbound deduplication (D-3), and the job
  runner as its own task (X6-6). The report length cap here keeps a report
  to one Signal part. It does not track parts.
- Reading Signal quote-replies. A quote-reply "yes" is treated as a plain
  "yes". Telling which preview a quote points at needs 018's record of send
  timestamps.
- Reactions. A thumbs-up reaction still has no text and is still ignored.
  Nothing runs, and nothing false is said.
- Re-checking stored bytes at send time. A relay body and a verbatim
  reminder sentence were shown to the person and confirmed or saved, so the
  runner still sends those bytes unchanged. They get the markup and empty
  checks once, when they are stored.
- The terminal REPL. It prints to a terminal, not to Signal.
- The rest of the eval harness gap (H-1). This spec builds only the replay
  and job-run parts it needs to prove itself.
- The separate bug fixes the brief excludes: D-1, D-7, D-11, G-1, G-2, G-3,
  G-5, H-4, H-5, A-5, F-5, F-7, X4-6, E-4, C-14.

## Design

### What 017 needs from 016

One function, owned by 016: its local-time formatter, called `format_local`
here as a placeholder for whatever name 016 gives it. 017 needs two things
from it:

- The canonical sentence for an instant relative to now, with an absolute
  date ("Saturday, October 31 at 8:00 AM"). Every server record below stores
  that form, so a stored record never says "tomorrow".
- The parts of that rendering: local date, hour, minute, and the day word
  016 would use ("today", "tonight", "tomorrow", a weekday). The time check
  in `check_outbound` compares the model's prose against these parts. It
  does not format or parse instants itself.

017 also needs every 016 tool result that states an instant (`job_create`,
`job_list`, `job_reschedule`, `relay`, `get_time`, and the calendar tools) to
carry it in one field shape that the turn's facts can collect. If 016 ships a
formatter that returns only a string, adding the parts is a change to 016's
module, agreed with that spec, not a second formatter here.

017 assumes no default hour. 016 has a daily job with no named hour ask the
person what time. That question is an ordinary reply: it passes
`check_outbound` because it is not a `yes_request`, and it stages nothing.
The person's answer is an ordinary turn. Every time 017 shows or checks
comes from a tool result, never from a default.

If 016 slips, 017's first deploy can still ship. The time check stays in
shadow mode until the formatter lands, and the 0.1.8 `_names_another_time`
regex stays in place until then.

### One outbound check: `src/assistai/outbound.py`

**Types.** Three frozen dataclasses say where a piece of outbound text came
from:

- `CheckedText`: model text that passed `check_outbound`. Only
  `check_outbound` constructs it.
- `ServerText`: text the gateway wrote (previews, summaries, notices, fixed
  failure sentences, the job message header and footer).
- `ConfirmedText`: bytes the person already confirmed or saved (a relay body,
  a verbatim reminder sentence).

`Outbound` is their union. `SignalChannel._safe_send`,
`SignalChannel.notify_job`, and `JobChannel.notify_job` take `Outbound`, not
`str`. mypy enforces the type. `tests/test_invariants.py` enforces the rest:
`CheckedText(` appears only in `outbound.py`, and `SignalTransport.send` is
called only from `SignalChannel._send`, `relay.deliver_stored_relay`, and the
update watcher.

**Inputs.** `check_outbound(text, facts, *, kind, settings) -> Verdict`.
`kind` is `reply`, `job_report`, or `preview_prefix` (the model's prose sent
above a preview). `facts` is a frozen `TurnFacts` built by
`facts_for_turn(messages, start, surface)` from the messages this turn
appended:

- `finish_reason` of the final completion. `run_turn` copies
  `Completion.finish_reason` onto the assistant `Message` it appends, in a new
  in-memory field `Message.finish_reason` that is not persisted. This is the
  missing link behind A-1.
- Each tool call this turn: name, whether it was staged, whether it ran, and
  whether its result had an `error` key.
- `surface.saved_reminders` and `surface.saved_job_ids`.
- The instants that tool results stated this turn, as 016's formatter parts.

**Checks.** Each check has a name, which is what the logs and settings use.

| Check | Refuses |
| --- | --- |
| `finish` | `finish_reason` of `length` or `content_filter` |
| `unknown_finish` | a missing `finish_reason` |
| `empty` | text that is empty after `strip_untrusted_wrappers` |
| `markup` | `<think>`, `</think>`, `<minimax:tool_call>`, `<invoke`, `<tool_call>`, `<\|` special tokens, or an untrusted wrapper that survived stripping |
| `impersonation` | the server's own wording: `_CONFIRM_HINT_ONE`, `_CONFIRM_HINT_MANY`, the relay hint, `_TAINT_NOTE`, the `format_executed` heading, a line starting `[job:`, or the gateway record frame |
| `length` | more than `job_report_max_chars` (1,900, one Signal part) for a report, or `outbound_reply_max_chars` (3,800) for a reply |
| `repetition` | text of at least 400 characters whose zlib ratio is under 0.25, or where one line of 10 or more characters makes up 30% of the lines and appears at least 4 times |
| `yes_request` | an explicit request to reply yes or confirm ("reply yes", "say yes", "reply 'yes'", "confirm by") in any model text, because the model cannot stage by asking |
| `action_claim` | a first-person claim that a staged-class action finished ("I archived", "I've sent", "cancelled it", "added to the calendar", "Done.", "All set.") when no matching call ran in this turn |
| `mail_claim` | a statement about the inbox ("unread", "new mail", "your inbox") in a reply when no mail tool succeeded this turn |
| `time_claim` | in a turn that called a job tool or `relay`, or saved a reminder, a clock time or day word in the prose that matches none of the instants the turn's tool results stated |

The impersonation strings are imported from `staging.py` and `jobs.py`, not
copied, so they cannot drift from what the server sends.

In a conversation turn, a staging tool never runs, except a verbatim
reminder. A model claim that such an action finished is therefore false by
construction, which is what makes `action_claim` safe to enforce. A job run
cannot stage at all, so in a report any action claim and any yes-request is
refused.

`time_claim` replaces `_names_another_time`, `_CLOCK_TIME`, and `_clock_key`
in `signal/channel.py`. The allow-list is the set of instants the tools
stated (the saved reminder, calendar events, `get_time`), compared by hour,
minute, and day word through 016's parts. It is no longer every clock string
inside the server's sentence. A dentist time from `calendar_today` beside a
reminder is kept. "Tuesday at 8" beside a stored Monday 8:00 AM, or "tonight"
beside a stored tomorrow, is caught. "8:55" beside "8:55 PM" matches. A time
inside the quoted reminder text is no longer treated as the stored time.

**Shadow mode.** `settings.outbound_shadow_checks` (env
`ASSISTAI_OUTBOUND_SHADOW_CHECKS`, comma-separated like `signal_allow_from`)
names checks that log `outbound.would_refuse` but do not refuse. The default
for the first deploy is
`repetition,unknown_finish,mail_claim,time_claim,reminder_words`, where
`reminder_words` is the B-1 check below. Everything else enforces from the
first deploy, because each of those refusals is either never legitimate
(`finish`, `empty`, `markup`, `impersonation`, `yes_request`) or a hard cap
the person cannot use anyway (`length`). `action_claim` enforces from day one
only in its narrow form: first-person past-tense verbs, and a whole reply of
"Done" or "All set". It enforces only after the offline replay below shows
no hit on a legitimate row. The last deploy empties the shadow list.

**Logging.** A refusal logs `outbound.refused` or `outbound.would_refuse`
with `check`, `kind`, `agent`, `chars`, and the first 12 hex digits of the
text's SHA-256. The text itself is never logged. Refused text can hold mail
subjects and page content.

### What happens when text is refused

**A conversation reply.** `_converse` removes the refused assistant row.
It then appends a repair note, a server row of kind `repair` that is never
persisted, and runs `run_turn` once more. The note says which check refused
the text and what to do instead. For `markup` that means calling the tool
properly. For `yes_request` it means calling the staging tool. For
`action_claim` it means not claiming the action ran. For `mail_claim` it
means calling `mail_inbox` first. For `length` it means writing something
shorter. Tool exchanges from either round are kept and persisted. Only the
refused text and the note are dropped. There is at most one repair round per
turn.

If the second answer is also refused, the person gets a fixed `ServerText`
sentence for that check, and history stores that sentence as a `notice`
record. The refused text is stored nowhere.

- `finish`: "My answer was cut off before it finished. Ask again, or ask for
  less at once."
- `empty`: "I did not get an answer from the model. Try again in a moment."
- `markup`, `repetition`: "My answer came out garbled, so I did not send it.
  Nothing was changed. Try again."
- `action_claim`, `yes_request`: the turn's preview if it staged, otherwise
  "I could not finish that. Nothing was changed. Ask again."
- `mail_claim`: "I need to open your inbox to answer that, and it did not
  work this time. Try again."
- `time_claim`: the server's reminder sentence alone, then "I left out the
  rest of my answer because it named a time I could not match to what was
  saved."

A blank completion is never stored, which closes A-3. The failure sentences
also say what failed. A tool-round cap now gets "That needed more steps than
I can take for one message. Try one thing at a time." It no longer gets "I
could not reach the model".

**A staged turn.** The phone message is the server's reminder sentences, if
a reminder was saved, then the checked prose, then the preview. Before this
spec, the prose was hidden from the phone but kept in history (X1-5, D-5).
Now the prose reaches the phone, after the same checks as any reply, so a
claim that the staged action already ran never sits above the preview. If
the prose is refused twice, the phone gets the preview alone, and the prose
is not stored. History stores exactly what the phone was sent. The reminder
sentence is stored once, as its own record (X2-8).

**A job report.** See the run contract below.

### The job-run contract: `src/assistai/jobrun.py`

**What a run is shown.** `job_view(agent, job, earlier_runs)` replaces
`trusted_context(messages)` in `SignalChannel.run_job`. A run sees:

1. `job_system_prompt(agent, job)`: identity, the clock line from 016's
   formatter, the report-only rules from `job_user_prompt`, and the output
   rule: "End with the report. If the tools found nothing this job asks
   about, reply with the single word NONE."
2. A trusted gateway record listing this job's last
   `settings.job_run_history` runs (default 3), newest first. Each entry has
   the run time, the tools that looked, their counts, and the status ("report
   sent", "nothing to report", "check failed"). Report text is not included.
   The run is never shown a report to copy.
3. A trusted frame: "Run the job '{name}' now."
4. The job's prompt as its own user message. When `job.tainted` is true that
   message is untrusted, so the nonce fence wraps it. The frame says the
   person confirmed it while outside text was in the conversation, and that
   it can only ask for a look and a report. This supersedes spec 015's "a
   tainted digest still gets its prompt as an instruction" (X4-4).

A run sees no conversation history. Not the person's lines, not relays, not
earlier digests. The surface's `Scope` is still taken from the full stored
history with `Scope.from_history`, so taint gating and labelling do not
change. Only what the model reads changes. `trusted_context` and its test
are deleted.

**What code decides.** `derive_run(scratch, start, job, surface) -> JobRun`
reads the run's own tool calls and results:

- `Look(tool, ok, count)` for each call to a tool in `REPORT_TOOLS`. `ok`
  means the result had no `error` key. `count` comes from
  `result_count(tool, content)`: the length of `messages` for `mail_inbox`
  (never `inbox_unread`), of `events` for `calendar_today`, and of `hits` for
  `web_search`. It is `None` for tools whose result has no count
  (`mail_read`, `web_fetch`, `shared_lists`). `shared_lists` returning "No
  lists yet." counts 0.
- `looked`: at least one `Look` with `ok`.
- `found`: any `ok` look with a count above 0, or with a count of `None`.
- `answer`: `nothing` when the final text, stripped of trailing `.` and `!`
  and casefolded, is exactly `none`. Otherwise `report`. The six-word
  `_NOTHING` set and `is_nothing` are deleted.
- `finish_reason` of the final completion, and the `check_outbound` verdict
  on the report text with `kind="job_report"`.

**One nudge, then a status.** Inside `run_job`, at most one extra round
runs. The cases are checked in order:

- If the run did not look, the round uses the existing no-look or
  tool-error variant of `look_again_prompt`.
- Otherwise, if the report was refused, the round uses the repair note for
  that check.
- Otherwise, if the job is a schedule, `found` is true, and the answer was
  NONE, the round says "The tools returned N items. Summarize them, or say
  NONE if none of them matter for this job." This replaces the listed-mail
  rule from spec 015.

Then `JobRun.status` is set:

| Status | When | Phone gets |
| --- | --- | --- |
| `reported` | looked, answer is a report, report accepted | `[job: name]`, the report, then the looks line |
| `nothing` | looked, answer NONE (a watch, or a schedule after the nudge) | schedule: `[job: name]`, "Nothing to report.", the looks line. Watch: nothing |
| `quiet` | a watch that looked and found nothing, whatever the text says | nothing |
| `not_checked` | did not look after the nudge | the fail notice |
| `refused` | the report was refused after the repair round | the fail notice with the reason ("the answer was cut off", "the answer was too long", "the answer came out garbled") |

The looks line is `ServerText` built by `format_looks`: "Looked at: mail (3
found), calendar (0 found)." It is the code's account of the look, so a
report cannot misstate whether the run checked.

`JobOutcome` is replaced by `JobRun`. `JobRunner._fire` branches on
`JobRun.status` instead of `outcome.found` and `outcome.checked`. A raised
exception still becomes the fail notice, as today. Report labelling is
unchanged: `untrusted = surface.fetched_untrusted or job.tainted`.

**Where runs are stored.** A new table, `job_runs`, written once when a run
finishes and before its message is queued:

```
CREATE TABLE IF NOT EXISTS job_runs (
    id TEXT PRIMARY KEY NOT NULL,
    job_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL NOT NULL,
    status TEXT NOT NULL,
    looks TEXT NOT NULL,
    finish_reason TEXT,
    refusal TEXT,
    report_chars INTEGER NOT NULL DEFAULT 0,
    untrusted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS job_runs_job ON job_runs(job_id, finished_at);
```

`Store.record_job_run(run)` inserts the row and, in the same transaction,
deletes all but the newest 20 rows for that job. `Store.recent_job_runs(agent,
job_id, limit)` filters by agent as well as job. Report text is not stored
here. It lives in the conversation record below, where the window ages it
out. Cancelled jobs keep their run rows for audit, like their job rows.

### Server text is stored as records, not as assistant speech

**Columns.** `messages` gains `origin TEXT NOT NULL DEFAULT 'legacy'` and
`record TEXT`. `Message` gains `origin: Literal["person", "model", "server",
"legacy"]` and `record: dict[str, object] | None`. Every `Message(` in
`src/` passes `origin` explicitly, and `tests/test_invariants.py` checks
that. Person rows are `person`. Rows `run_turn` appends are `model`. Rows the
gateway writes are `server` and carry a record. Server rows keep role
`assistant` (role `user` for a relay record, as today), so the window, the
count trim, and a downgrade behave as before. `content` stays the exact text
the phone was sent.

**Records.** New module `src/assistai/records.py` builds them. Each record is
JSON with `kind`, `ref`, and `text`. `text` is a third-person sentence for
the model, written with 016's formatter where it names a time.

| Kind | Written by | `text` says |
| --- | --- | --- |
| `preview` | `_converse` when a turn stages | the server showed a proposal to do X (from `staging.describe_calls`, third person, no confirm hint), and it runs only if the person's next message is yes |
| `decision` | `_handle_decision`, and a turn that withdraws | the person said yes and it ran, said no and it was discarded, answered after it expired, or sent another message and it was withdrawn. In every case except yes, nothing ran |
| `commit` | `_commit_proposal` | what ran and each call's result, from the same results `format_executed` uses |
| `reminder_saved` | `_converse` when a reminder saved | the job name and id, the text, and the first run as an absolute local date and time |
| `job_run` | `persist_job_report` | the job name, run time, looks, status, and the report the person was sent, clipped to 1,500 characters |
| `job_notice` | `persist_job_report` | a fail or expire notice for the named job |
| `reminder_sent` | `persist_job_report` | the reminder text that was texted, and when |
| `relay_sent` | `persist_job_report` | the relay that went to the other phone, and when |
| `relay_record` | `_append_untrusted` | who relayed what, on the recipient's side |
| `notice` | `_converse`, `_handle_decision` | a fixed failure sentence the person was sent |

`jobs` gains `pending_record TEXT`. `JobRunner._queue` stores the record to
persist next to `pending_text`, so the scheduler, not the channel, says what
kind of message went out. `persist_job_report(agent, text, *, record,
untrusted)` takes it. 018 later replaces this outbox, so the column is kept
as small as possible.

**What the model sees.** The serializer in `inference/types.py`,
`wire_messages(messages, nonce)`, which `FireworksClient.complete` calls,
renders the rows:

- `person` and `model` rows go out as today, with one change. A `model`
  assistant row flagged untrusted is now wrapped in the nonce fence like any
  other untrusted row (A-4). Before this, assistant rows were never wrapped.
  The model copying those tags back is caught by `strip_untrusted_wrappers`
  and the `markup` check.
- A `server` row goes out as role `user` with the frame "Gateway record,
  written by the server and not by you or the person, {local time}:" followed
  by `record["text"]`. An untrusted server row is fenced like any other.
- A `legacy` assistant row with no tool calls is checked against the shapes
  the server used to write: it starts with `[job:`, ends with a confirm hint,
  or starts with `Ran ` and an action count. If it matches, it is left out of
  the request. Legacy rows that fail `repetition` or exceed 4,000 characters
  are also left out. Other legacy rows go out as assistant text, as before.
  `history.legacy_omitted` logs the count per load.

Leaving a row out of the request never changes what is stored, so
`Scope.from_history` still sees its untrusted flag, and taint ages out on
the same schedule as before. Untrusted stays sticky. `last_assistant_text`
skips `server` rows.

The model learns from the system prompt that gateway records are written by
the server, that it cannot write one, and that a proposal exists only when a
`preview` record says so. The `agents.py` sentences that stood in for this
("Do not ask them to reply yes in your own text", "If an earlier reply said
that, it was wrong") stay until the eval cases below pass without them. The
`impersonation`, `yes_request`, and `action_claim` checks are what enforce
the rule.

### What a yes is bound to

`proposals` gains `id TEXT`. `Proposal` gains `id`, and the `preview` record's
`ref` is that id.

- **A yes or no applies only to the preview it answers.** `_handle_decision`
  commits or discards only when the newest stored row for that person is the
  `preview` record whose `ref` equals the live proposal's `id`.
- **Any other person message withdraws the proposal.** If a proposal is live
  and the inbound text is not a decision, `_converse` clears it before the
  turn runs. It then appends a `decision` record ("withdrawn, nothing ran")
  ahead of the person's row, and logs `staging.withdrawn`. This is what
  `architecture.md` already says ("The next inbound message either confirms
  it or does not"). It covers "yes but say 7", "what time is that?", and the
  archive-in-prose turn (D-2, B-3). The model sees the withdrawal and can
  stage the corrected call. The relay preview's "or say what to change" still
  works the same way: a critique re-stages.
- **A server message in between re-shows the preview.** If a job report, a
  fired reminder, or a relay from the other person was stored after the
  preview, a yes does not run anything. The server sends the preview again
  with the lead "Another message arrived after this, so that yes did not run
  it." It appends a fresh `preview` record with the same `ref`, and logs
  `staging.reshown`. The next yes then applies. The expiry does not move.
- **Near-yes words.** `parse_decision(text, calls)` casefolds the message,
  trims whitespace and trailing punctuation and emoji, and matches whole
  messages against `CONFIRM_WORDS` and `REJECT_WORDS`. The default confirm
  set is yes, y, yeah, yep, yup, ok, okay, sure, confirm, do it, go ahead,
  send it, yes please, and 👍 sent as text. The default reject set is no, n,
  nope, discard, and cancel. These are the owner's sets (question 1).
  Anything else is not a decision. A 👍 reaction has no text and is not a
  decision.
- **"cancel" on a cancellation.** When every stored call is `job_cancel`,
  "cancel" and "stop" are ambiguous. `parse_decision` returns `ambiguous`.
  The server re-shows the preview with "Reply yes to cancel {name}, or no to
  keep it." and logs `staging.cancel_clarified` (B-9).
- **A decision with nothing live.** If a decision-shaped message arrives,
  no proposal is live, and the newest server record is a `preview` or
  `decision`, the server answers with a fixed sentence that says what
  happened. For example: "Nothing is waiting for a yes. That proposal
  expired, so nothing ran. Ask again if you still want it." It stores that
  as a `decision` record and logs `staging.expired_yes`. A second yes after
  expiry therefore never reaches the model as a reply to a live-looking
  preview (X2-4). A decision-shaped message with no recent proposal (an "ok"
  after a job report) is an ordinary turn, and the outbound check stops any
  "Done" in the answer.

Before the schema change ships, the first deploy binds by exact content. A
preview row is the newest row, and its content equals
`format_proposal(pending.calls, tainted=pending.tainted)`, or ends with it
when a reminder sentence came first. That comparison is against text the
server wrote, not a heuristic. The second deploy replaces it with the `ref`.

### Smaller fixes in the same paths

- **X4-1.** `BoundSurface.execute` returns the staged status result with
  `untrusted=self._scope.tainted`. Outside text copied into a staged call's
  arguments keeps the fence on the next turn.
- **B-1.** `_reminder_saves_now(call, person_text)` also requires that the
  stored sentence is the person's own request. At least 60% of its content
  words (three or more letters, not stopwords, casefolded, with a trailing s,
  ed, or ing removed, and ignoring "don't", "forget", "remember", "reminder")
  must appear in the person's message this turn. Otherwise the reminder
  stages, and the person sees its text before it is saved.
  `BoundSurface` gets the person's text from `ToolBroker.for_agent`.
  [Decision 0003](../decisions/0003-reworded-reminders-wait-for-yes.md)
  accepts this rule. It ships log-only, as the `reminder_words` shadow check
  (`broker.reminder_unmatched` with the overlap ratio, and the reminder
  still saves). It enforces only in the commit that also amends the PRD
  § Jobs sentence (Rollout step 15).
- **Stored bytes.** `jobs.parse_create` and `relay.parse_body` refuse text
  that fails the `markup` check, so the bytes that are later sent unchanged
  were clean when stored.
- **F-2, the tool side.** The `mail_inbox` description and the `agents.py`
  mail sentence say that `messages` and `matches` are what the search found,
  and that `inbox_unread` is the whole inbox's unread total, not the search
  result. `_tool_listed_mail` is deleted. The result shape does not change.

### Schema 16 and its migration

`SCHEMA_VERSION` is 16. The migration from 15 runs in one explicit
transaction (`BEGIN IMMEDIATE` ... `COMMIT`, not `executescript`, which
commits first), so a SIGTERM mid-migration leaves a version 15 file intact.
It does the following:

- Adds `messages.origin` and `messages.record`. It sets `origin` to `person`
  for user rows with `untrusted = 0`, and to `server` with a `relay_record`
  record built from the content for user rows with `untrusted = 1`. Tool
  rows and assistant rows with tool calls become `model`. Assistant rows
  without tool calls stay `legacy`. It does not touch `untrusted`,
  `created_at`, `content`, or `seq`.
- Adds `proposals.id` and gives any live row a random id. Such a row has no
  `preview` record, so a yes to it re-shows the preview first.
- Adds `jobs.pending_record`, null for every row, active or cancelled. A
  pending report with no record is persisted as a `job_notice` record built
  from its text.
- Creates `job_runs`. It does not change any other `jobs` column, so 016's
  schedule fields and the live rows (the active "mgmplus cancel reminder" and
  the cancelled rows kept for audit) come through byte for byte.

A file at version 14 migrates through 016's step to 15 and then to 16.

**Downgrade.** 016 adds the `python -m assistai store` command group and its
`store downgrade` command. 017 adds that command's 16-to-15 step, which runs
for `python -m assistai store downgrade --to 15`. The step drops
`job_runs`, `jobs.pending_record`, `proposals.id`, `messages.origin`, and
`messages.record` with `ALTER TABLE ... DROP COLUMN`, which needs SQLite
3.35 or later. It then sets `user_version` to 15, all in one transaction.
Server rows keep `content`, which is the phone text, so the previous image
reads them as the assistant rows it would have written itself.

### Config keys

All live in `Settings`, with the `ASSISTAI_` prefix:

- `outbound_shadow_checks: str`, default
  `"repetition,unknown_finish,mail_claim,time_claim,reminder_words"`.
- `outbound_reply_max_chars: int`, default 3,800.
- `job_report_max_chars: int`, default 1,900.
- `job_run_history: int`, default 3.

### Docs that change with the code

- `architecture.md`: § Staged actions (binding, near-yes, withdrawal,
  re-show); § Jobs (the run contract, no conversation context, the looks
  line, NONE handling, no listed-mail rule); § Persistence (`job_runs`,
  origins and records); § Prompt injection defenses (untrusted assistant
  rows fenced, gateway records).
- `docs/pi-deploy.md`: the pre-deploy volume tarball and the downgrade
  command under "Install a later image".
- `docs/release-notes.md`: what the household will notice.
- Spec 015 is superseded on three points: the tainted digest prompt is now
  fenced, the listed-mail rule is replaced by the counted nudge, and the
  reminder-time regex is replaced by `time_claim`. The architecture says so.
- `prd.md` § Jobs: the "remind me" sentence is amended to match decision
  0003, in the commit that turns B-1 enforcement on (Rollout step 15), and
  not before. `architecture.md` § Jobs changes in the same commit.
- `docs/decisions/0003-reworded-reminders-wait-for-yes.md` lands with this
  spec. The decisions index row for it is added when the spec merges.

## Risks and mitigations

**A job that drew on conversation context loses it.** The audit names this
risk. Today `trusted_context` gives a run the person's unanswered lines and
earlier no-tool reports, and nothing else once the window is tainted
(X2-2). That is mostly what caused the echo. No PRD text gives a job the
conversation. Memory and relay promise that job output and relays reach the
conversation, and they still do. The only active job on the Pi is a verbatim
reminder, which runs no model. Mitigation: the job's prompt has to be
self-contained, and the `job_create` description says so. The run gets its
own earlier run times, so "new since last time" still works through a Gmail
`after:` search. The owner confirmed that a job sees only its prompt and its
own earlier run facts (question 3).

**A strict check suppresses a real message.** The audit names this risk too.
Mitigations:

- *Shadow first.* Every check that judges wording (`repetition`,
  `mail_claim`, `time_claim`, `reminder_words`, the broad `action_claim`
  form) starts in shadow. It moves to enforce only after a week of
  `outbound.would_refuse` logs on the Pi and a clean offline replay.
- *An offline replay, no tokens.* `uv run python -m evals.replay --db PATH`
  runs `check_outbound` and the legacy-row rules over every assistant row and
  pending report in a copy of the Pi database. It prints counts per check
  with row ids. The copy stays outside the repo and is deleted afterwards. A
  check moves to enforce only when its hits on rows the owner marks
  legitimate are zero, and its hits on the four echo rows and the two
  model-written previews (jacob rows 11 and 22) are all of them. This replay
  runs before every deploy in the rollout.
- *A model replay, with tokens.* The job-mode eval cases below run the real
  runner twice over real-shaped history and grade the outbound properties,
  not tool names.
- *Refusal costs a repair round, not the message.* A refused reply gets one
  retry with the reason. A refused report gets one retry, then a fail notice.
  Nothing is dropped silently, which keeps the PRD's "failures never fail
  closed with silence".

**The wording checks are regexes over model text, which is the pattern the
audit faults.** True for `action_claim`, `yes_request`, `mail_claim`, and
`time_claim`. They are not what this spec relies on. The binding, the
records, the run contract, and the token-cap and markup checks are
structural. The wording checks are a second layer. They live in one module,
are tested against the property rather than against the format they match,
and on a hit they trigger a repair round instead of quietly dropping text.
`action_claim` is narrow on purpose: in a conversation turn, a staged-class
action cannot have run, so the claim is false whenever the words match.

**Gateway records as user-role messages confuse the model or the provider.**
Consecutive user messages and third-person records are new for minimax-m3.
Mitigations: `tests/test_live.py::test_gateway_records_are_accepted` sends a
history with records through real Fireworks before the second deploy. The
eval cases below include records in history. If the provider rejects
consecutive user messages, `wire_messages` merges adjacent user-role rows.
That is a serializer change, not a storage change.

**Fencing untrusted assistant rows makes the model copy tags.** Mitigation:
`strip_untrusted_wrappers` already runs on every completion, and the
`markup` check refuses a wrapper that survives. The eval case
`untrusted-digest-follow-up` measures how often tags come back before the
second deploy. If they come back often, the fence can be swapped for a
one-line "earlier reply written with outside text in context" marker
without changing storage.

**The yes binding annoys.** A yes after a job report now re-shows the
preview, and any other message withdraws it. Mitigations: re-show and
withdrawal both say plainly what happened. `staging.withdrawn` and
`staging.reshown` are logged, so the owner can see how often it happens in
the first week. The household is told before the second deploy.

**A wider confirm set confirms by accident.** "ok" right after a preview now
runs it. Mitigation: a decision applies only to the preview that is the
newest stored row, so "ok" can only answer what the person was just shown.
The owner chose the word set.

**More model calls lengthen turns across a deploy.** A repair round makes a
turn take longer, and so widens the window in which SIGTERM cancels a turn.
The cancellation itself is 018's. Mitigation: at most one repair round per
turn, and deploys happen when nobody has just messaged (rollout steps
below).

**The migration misclassifies rows.** Mitigations:

- `tests/real_shaped.py` builds a version 15 file shaped like the Pi's. It
  has jacob's 30 rows: four 8,510-character echo reports, the two
  model-written previews ending in the confirm hint, the "Cancelled morning
  email summary." row, bare "Yes" lines, and a relay record. It has spouse's
  rows, the active mgmplus repeating verbatim reminder (every 86,400 seconds,
  first run Dec 10 at 8:00 CST), the cancelled morning email job, and a live
  proposal.
- The migration test checks row counts, untrusted flags, `Scope.from_history`
  per agent, and every `jobs` row before and after.
- Before the second deploy, the migration runs on the Mac against a copy of
  the Pi's volume, and the same comparisons are made by hand.
- `legacy` rows are only left out of requests, never deleted, so a wrong rule
  is fixed in code without data loss.

**An older image cannot open the newer database.** `_init_schema` refuses a
file whose version is above its own, so rolling back to the 016 image after
the second deploy would not start. Mitigations: the downgrade command above,
tested in `tests/test_store.py`. Before the deploy, the volume is saved as a
tarball, the same way `docs/pi-deploy.md` copies it (`alpine tar czf`). The
tarball is the backstop if the downgrade fails. Using it loses only what
happened after the deploy.

**Pinned tests have to change.** Each change below is intended, because each
test pins behaviour this spec fixes. These are the tests from the audit's
section 6 that this spec owns:

- `test_jobs.py::test_schedule_run_messages_the_owner`,
  `::test_cancel_after_send_still_saves_history`, and
  `::test_job_report_is_not_tainted_by_older_history` expect a no-tool answer
  to be texted and stored trusted (C-5). Their fixtures gain a real look,
  and a new test asserts the no-tool case is a failed check.
- `test_conversation.py::test_trusted_context_drops_untrusted_without_splitting_tools`
  expects the job context to keep the no-tool `[job: mail] quiet` line. It is
  deleted with `trusted_context`.
- `test_research.py::test_an_assistant_reply_is_not_wrapped_again` requires an
  untrusted assistant row to go out unwrapped (A-4). It becomes a test that
  an untrusted assistant row is fenced and a trusted one is not.
- `test_staging.py::test_only_a_whole_message_yes_confirms` asserts "cancel"
  is always a reject (B-9) and "yes please" is not a confirm (X1-3). It keeps
  "cancel" as a reject except on a `job_cancel` proposal and takes the
  approved word sets.
- `test_channel.py::test_failed_turn_leaves_no_dangling_tool_call` requires
  "I could not reach the model" for a tool-round cap (the A-3 family). It
  expects the round-cap sentence instead.

Other existing tests that change for the same reasons:
`test_channel.py::test_a_reply_naming_another_time_is_caught` moves to the
`time_claim` cases, and
`test_channel.py::test_a_staging_tool_shows_the_stored_call_not_model_prose`
now expects checked prose above the preview. `test_jobs.py`'s
`_FakeJobChannel` returns a `JobRun`. New job tests drive the real
`SignalChannel.run_job` with scripted completions from `tests/fakes.py`, so
the real derivation runs. Hand-written outcomes are what let C-5 and F-2
through before.

**Tests that must exist and fail before the change.** The first commit adds
them as `pytest.mark.xfail(strict=True)`, each with the finding id in the
reason. Each fixing commit removes its marker. They are the audit's missing
tests 1 and 5: the replayed-history job run, and the proposal-binding
conversation tests (AC1 to AC6 and AC13 to AC17 below).

**What to watch on the Pi.** After each deploy, run `docker compose logs
gateway` and look for these events:

- `outbound.refused` and `outbound.would_refuse`, by `check` and `kind`. Any
  `impersonation` refusal is worth reading. `would_refuse` counts decide the
  last deploy.
- `outbound.repaired` and `outbound.gave_up`: how often the repair round
  saves a reply.
- `jobs.run_recorded`, with `status`, `looked`, `found`, `finish_reason`, and
  `report_chars`. Any `not_checked` or `refused` on a real digest is worth
  reading.
- `staging.withdrawn`, `staging.reshown`, `staging.cancel_clarified`,
  `staging.expired_yes`, and the existing `staging.committed`.
- `history.legacy_omitted`, which should fall to zero as the window ages out
  the old rows, by about Nov 13.
- `store.migrated` once, with from 15 to 16 and row counts. `store.downgraded`
  only on a rollback.
- `broker.reminder_unmatched`, with its ratio, while B-1 is in shadow.

**The household is told.** Jacob tells his spouse himself, by his own
message and not through the bot, before the second deploy: "From tonight, a
proposal only runs if yes (or ok, sure, 👍) is the next thing you send. If
you send anything else first, it drops the proposal and you ask again. If a
job report arrives in between, it shows you the proposal again. Job messages
end with a line saying what was checked." The release notes say the same.
Nothing is announced by the bot, because the bot does not send unprompted
messages except jobs.

## Rollout

Three deploys. Each one ships and rolls back on its own. Commit messages
follow `AGENTS.md`.

**Deploy A: the output contract and the job-run contract, no schema change.**
It needs 016 on the Pi.

1. "Pin the job echo, the impersonated cancel, the stale yes, and the empty
   search as failing tests before fixing them." This adds the strict-xfail
   tests, `tests/real_shaped.py`, the job-mode eval harness, and the eval
   cases.
2. "Carry the final finish_reason onto the assistant message so a cut-off
   answer can be refused."
3. "Send model text to Signal only through check_outbound, so a cut-off,
   empty, marked-up, or impersonating answer never reaches a phone." This
   adds the types, the invariant tests, and the shadow settings.
4. "Give a refused reply one repair round and a fixed sentence that says
   what failed, and never store a blank answer."
5. "Decide looked and found from the run's own tool results, and show a job
   its prompt instead of the conversation." This adds `jobrun.py`, deletes
   `trusted_context`, `_NOTHING`, and `_tool_listed_mail`, fences the
   tainted prompt, adds the looks line, and corrects the `mail_inbox`
   description.
6. "Label a staged result untrusted when the turn is tainted, so copied mail
   text keeps its fence."
7. "Run a yes only when it answers the preview just shown, and accept the
   approved near-yes words." This uses binding by exact content, withdrawal,
   re-show, and cancel clarification.
8. Docs: architecture § Jobs and § Staged actions, and release notes.

Before deploying, run the offline replay on a fresh copy of the Pi database
and confirm the enforced checks have no hit on a legitimate row. Deploy when
`SELECT count(*) FROM proposals` is 0 and nobody has messaged in the last
few minutes. Roll back with the previous image tag. There is no schema
change.

Manual checks that close deploy A, from a phone:

- Ask "anything in my inbox?" and get a normal reply.
- Create "every 5 minutes, check my inbox for anything from school", confirm
  it, and get one report under 1,900 characters. It must end with a "Looked
  at: mail" line that matches what the inbox holds. Then cancel it by
  replying "cancel" to the cancel preview, see the clarification, and reply
  yes.
- Stage a list change, send "wait", and see it withdrawn. Stage it again,
  reply "Yes!", and see it run.
- `sqlite3` shows the mgmplus row unchanged.
- The logs show `jobs.run_recorded status=reported` for the test job.

**Deploy B: records, prior runs, and schema 16.**

9. "Store server text as labelled records at schema 16, with a downgrade to
   15." This adds the migration, the 16-to-15 step in 016's `store
   downgrade` command, and the store tests.
10. "Show the model gateway records instead of server text in its own voice,
    and leave old echo and preview rows out of the request." This adds
    `wire_messages`, the `legacy` rules, absolute dates, and fencing for
    untrusted assistant rows.
11. "Record each job run and show the next run its earlier results."
12. "Bind a yes to its preview by id, and record expired and withdrawn
    proposals."
13. Docs: architecture § Persistence and § Prompt injection defenses,
    `pi-deploy.md` downgrade steps, and release notes.

Before deploying: run the offline replay, `make live`, and `make eval`. Run
the migration on the Mac against a copy of the Pi's volume, and compare row
counts, untrusted flags, and the `jobs` table. Tar the Pi's state volume.
Confirm `proposals` is empty. Tell the household. Roll back with `docker
compose run --rm --no-deps gateway python -m assistai store downgrade --to
15`, then the previous image tag. If that fails, restore the tarball.

Manual checks that close deploy B:

- `PRAGMA user_version` is 16, and the per-agent row counts match the
  pre-deploy copy.
- `store.migrated` appears once, and `history.legacy_omitted` reports jacob's
  echo rows.
- Ask "what was in the last email summary?" and get an answer that treats it
  as a record, not as its own words.
- Stage something, wait for any job message to arrive, reply yes, and see
  the preview re-shown.

**Deploy C: enforce the tuned checks.** No image change if the defaults stay
as they are.

14. After a week of `outbound.would_refuse` logs and a clean replay, set
    `ASSISTAI_OUTBOUND_SHADOW_CHECKS=reminder_words` in `.env` on the Pi and
    restart. Then a commit makes that list the default: "Enforce the
    repetition, inbox, and time checks now that a week of shadow logs shows
    no real message would be refused." Once `time_claim` enforces, the same
    commit deletes `_names_another_time`. `reminder_words` stays log-only,
    because enforcing it changes PRD behaviour.
15. After a week of `broker.reminder_unmatched` logs shows the 60% threshold
    stages no reminder the person phrased themselves: "Wait for a yes when a
    reminder's sentence is not the person's own words, as decision 0003
    accepts." This commit empties the default shadow list, amends the PRD
    § Jobs "remind me" sentence, and updates architecture § Jobs and the
    release notes, all together. Then `.env` on the Pi sets the empty list,
    and the household is told that a reminder the assistant rewords now
    shows its text and waits for a yes.

Roll back step 14 by restoring the shadow list in `.env`. Roll back step 15
by reverting its commit, which also reverts the PRD sentence, so the code
and the PRD never disagree. Setting `reminder_words` in `.env` alone is only
a stopgap until that revert ships. Manual checks: a week with no
`outbound.refused` on a message the owner wanted to receive, and after step
15, "remind me to take the bins out at 7" saves at once, while "every day at
7, check my email and summarize it" never saves a verbatim reminder without
a yes.

## Acceptance criteria

Each criterion names the test, eval case, or manual step that proves it, and
the findings it closes.

**The output contract**

- [ ] AC1 — A job report whose final completion stopped at `length` is not
  sent or stored. The run is `refused`, and the owner gets the cut-off fail
  notice (`tests/test_jobs.py::test_a_report_cut_off_at_the_token_cap_is_not_sent`).
  The same holds for a reply, which gets one repair round and then the fixed
  sentence (`tests/test_channel.py::test_a_reply_cut_off_at_the_token_cap_is_not_sent`).
  The final `finish_reason` reaches the caller
  (`tests/test_loop.py::test_the_final_finish_reason_reaches_the_caller`).
  Closes A-1.
- [ ] AC2 — An empty completion is not stored, gets one repair round, and
  then the empty-answer sentence. A tool-round cap gets its own sentence
  (`tests/test_channel.py::test_an_empty_completion_is_not_stored`,
  `tests/test_channel.py::test_a_round_cap_failure_says_so`). Closes A-3.
- [ ] AC3 — Tool-call or thinking markup in content is not sent, in a reply
  or a report. A repair round that makes the real call runs it
  (`tests/test_channel.py::test_tool_markup_in_content_is_not_sent`,
  `tests/test_jobs.py::test_tool_markup_in_a_report_is_not_sent`). Closes
  X1-7.
- [ ] AC4 — A report over 1,900 characters, or one that repeats a line, is
  repaired or refused, and is never stored as a report
  (`tests/test_jobs.py::test_a_report_over_the_cap_is_not_stored`,
  `tests/test_jobs.py::test_a_repeating_report_is_refused`, the latter run
  with `repetition` enforced). Closes X2-6 with AC11.
- [ ] AC5 — The Oct 29 exchange fails at both steps. A reply that writes the
  server's preview wording with nothing staged is refused, and so is a reply
  that writes "Ran 1 action" or "Cancelled morning email summary." with no
  `job_cancel` committed
  (`tests/test_channel.py::test_a_model_written_preview_and_summary_are_refused`).
  At the model level, "Stop the email summary, I need to fix this bug" after
  a `job_list` result stages `job_cancel`, and the reply contains no confirm
  hint (`evals/cases/stop-the-job-stages-a-cancel.toml`). Closes the
  Oct 29 impersonated cancel.
- [ ] AC6 — "Done" or "archived" with nothing run is not sent. After a
  "Yes!" that reaches the model with no live proposal, the reply claims
  nothing (`tests/test_channel.py::test_a_done_claim_with_nothing_run_is_not_sent`).
  Closes X1-3 with AC15.
- [ ] AC7 — In a turn that saved a reminder and staged a call, the phone
  gets the server sentence, then checked prose, then the preview. Prose that
  claims the staged call ran is refused, and the reminder sentence is stored
  once
  (`tests/test_channel.py::test_prose_claiming_a_staged_action_ran_is_not_shown_above_the_preview`,
  `tests/test_channel.py::test_a_reminder_and_a_staged_call_store_the_sentence_once`).
  Closes X1-4 for claims that the action ran, and the duplicate X2-8. A
  different count in the prose ("4 newsletters" against a 3-message
  preview) is narrowed by the preview below it, not closed.
- [ ] AC8 — In a staged turn, the answers to the other asks reach the phone
  above the preview, and history holds exactly what was sent
  (`tests/test_channel.py::test_a_staged_turn_sends_the_other_answers_and_stores_only_what_was_sent`).
  Closes X1-5.
- [ ] AC9 — An answer about the inbox with no mail tool call this turn gets
  one repair round that calls `mail_inbox`
  (`tests/test_channel.py::test_an_inbox_answer_without_a_mail_call_is_asked_to_look`
  with `mail_claim` enforced, and
  `evals/cases/mail-answer-needs-a-look.toml`). Closes B-10.
- [ ] AC10 — In a reminder turn, a reply naming the wrong day ("tonight"
  against a stored tomorrow, "Tuesday" against Monday) is repaired. A reply
  naming a calendar event's time beside the reminder is kept, and so is
  "8:55" beside "8:55 PM"
  (`tests/test_channel.py::test_a_reminder_reply_naming_the_wrong_day_is_repaired`,
  `tests/test_channel.py::test_a_calendar_time_beside_a_reminder_is_kept`,
  both with `time_claim` enforced and 016's formatter). Closes X3-2.

**History and the job-run contract**

- [ ] AC11 — A job report is stored as a `job_run` record and reaches the
  model as a gateway record, not as assistant speech. Legacy `[job:`,
  confirm-hint, and "Ran N actions" rows are left out of the request while
  their untrusted flags still count toward `Scope`
  (`tests/test_channel.py::test_a_job_report_is_stored_as_a_record_not_assistant_speech`,
  `tests/test_conversation.py::test_legacy_echo_rows_are_not_sent_to_the_model`,
  `tests/test_conversation.py::test_omitted_legacy_rows_still_taint_the_scope`).
  Closes X2-1.
- [ ] AC12 — A job run sees its system prompt, its earlier runs' facts, and
  its prompt, and nothing from the conversation: no person lines, no
  relays, no earlier reports
  (`tests/test_jobs.py::test_a_job_run_sees_its_earlier_runs_not_the_conversation`).
  Closes X2-2.
- [ ] AC13 — The replayed-history job eval passes. Starting from jacob's
  real-shaped rows, the morning email job runs twice through the real runner
  with a different subject each time. Each report names its own subject,
  stays under 1,900 characters, contains no `[job:` line, and is `refused`
  when the scripted fixture ends at `length`
  (`evals/cases/replay-echo-job.toml`, plus its offline twin
  `tests/test_jobs.py::test_replayed_echo_history_does_not_echo`). Closes
  A-1, X2-1, and X2-6 at the model level.
- [ ] AC14 — A job answer that is a sentence with no tool call is a failed
  check, not a report. A watch whose tools found nothing stays quiet,
  whatever it says. Both use the real `run_job`
  (`tests/test_jobs.py::test_a_sentence_answer_with_no_tool_call_is_a_failed_check`,
  `tests/test_jobs.py::test_a_watch_whose_tools_found_nothing_stays_quiet_whatever_it_says`).
  Closes C-5 (with C-6 and X4-3).
- [ ] AC15 — A search that matches nothing while the inbox has unread mail
  is "Nothing to report" with "mail (0 found)", not a failed check. NONE
  after calendar events gets one nudge and then shows the count
  (`tests/test_jobs.py::test_an_empty_search_with_an_unread_inbox_is_nothing_to_report`,
  `tests/test_jobs.py::test_none_after_calendar_events_is_nudged_and_shows_the_count`,
  `evals/cases/job-search-empty-inbox-unread.toml`). Closes F-2 and C-13.
- [ ] AC16 — A tainted job's prompt reaches the run fenced as untrusted. A
  prompt that says "do not call tools" produces a failed check, not its text
  (`tests/test_jobs.py::test_a_tainted_jobs_prompt_is_fenced_in_the_run`).
  Closes X4-4.
- [ ] AC17 — An untrusted assistant row and an untrusted server record are
  fenced on the wire, and a trusted assistant row is not. An echoed wrapper
  in a completion is stripped and not sent
  (`tests/test_research.py::test_an_untrusted_assistant_reply_is_fenced_on_the_wire`,
  `tests/test_channel.py::test_an_echoed_wrapper_is_not_sent`,
  `evals/cases/untrusted-digest-follow-up.toml`). Closes A-4 (with X4-2 and
  X2-7).
- [ ] AC18 — A saved reminder's record and every server record name an
  absolute local date through 016's formatter. A record read the next day
  does not say "tomorrow"
  (`tests/test_channel.py::test_a_saved_reminder_record_has_an_absolute_date`).
  Closes X2-5.
- [ ] AC19 — A staged result in a tainted turn is stored untrusted and fenced
  on the next turn
  (`tests/test_broker.py::test_a_staged_result_in_a_tainted_turn_is_untrusted`).
  Closes X4-1 (with F-1).

**What a yes is bound to**

- [ ] AC20 — An ordinary turn withdraws the live proposal. A later "yes"
  after the model asked in prose runs nothing and says so. "yes but say 7"
  never sends the 6
  (`tests/test_channel.py::test_an_ordinary_turn_withdraws_the_live_proposal`,
  `tests/test_channel.py::test_a_yes_after_a_prose_question_runs_nothing`,
  `tests/test_channel.py::test_a_correction_does_not_send_the_old_draft`).
  Closes D-2 (with X1-2, X2-3, and B-3).
- [ ] AC21 — A yes after a job report, a fired reminder, or a relay record
  re-shows the preview, and the next yes runs it
  (`tests/test_channel.py::test_a_yes_after_a_job_report_reshows_the_preview`).
  Closes D-2's interleaving case.
- [ ] AC22 — "Yes!", "ok", "sure", "yes please", and 👍 as text confirm a
  live preview, and "ok wait" does not
  (`tests/test_staging.py::test_near_yes_words_confirm`,
  `tests/test_channel.py::test_yes_with_punctuation_runs_the_preview`).
  Closes X1-3 with AC6.
- [ ] AC23 — "cancel" in reply to a `job_cancel` preview re-shows it with a
  yes-or-no question and does not discard it. "cancel" on any other preview
  still discards
  (`tests/test_staging.py::test_cancel_on_a_cancel_job_proposal_is_ambiguous`,
  `tests/test_channel.py::test_cancel_reply_to_a_job_cancel_preview_keeps_it_live`).
  Closes B-9.
- [ ] AC24 — An expired yes is recorded. A second yes gets the "nothing is
  waiting" sentence and does not reach the model
  (`tests/test_channel.py::test_an_expired_yes_is_recorded_and_a_second_yes_runs_nothing`).
  Closes X2-4.
- [ ] AC25 — While shadowed, a reminder whose sentence is not the person's
  words saves and logs `broker.reminder_unmatched`. Once enforced, it stages,
  and a digest marked verbatim with the prompt "No new mail." waits for a yes
  (`tests/test_broker.py::test_a_reminder_whose_text_is_not_the_persons_is_logged`,
  `tests/test_broker.py::test_a_reminder_whose_text_is_not_the_persons_stages`).
  A reminder whose sentence is mostly the person's words still saves at once
  in a clean conversation
  (`tests/test_broker.py::test_a_reminder_in_the_persons_words_saves_at_once`).
  Enforcement lands in the same commit as the PRD § Jobs amendment that
  decision 0003 requires (Rollout step 15, manual read of `prd.md`). Closes
  B-1.

**Store, invariants, rollout**

- [ ] AC26 — A real-shaped version 15 file opens at 16. Row counts,
  `untrusted` flags, `Scope.from_history` per agent, and every `jobs` row are
  unchanged, and a live proposal gets an id. A version 14 file reaches 16
  through 15. A failure mid-migration leaves version 15 intact
  (`tests/test_store.py::test_schema_v15_gains_history_origins_and_keeps_rows`,
  `tests/test_store.py::test_schema_v14_upgrades_through_15_to_16`,
  `tests/test_store.py::test_a_failed_migration_leaves_v15_intact`).
- [ ] AC27 — The downgrade leaves a version 15 file whose tables match the
  016 schema exactly, with server rows holding their phone text. The
  gateway image's SQLite supports `DROP COLUMN`
  (`tests/test_store.py::test_downgrade_to_15_matches_the_016_schema`,
  `tests/test_store.py::test_sqlite_supports_drop_column`). On the Mac, the
  016 image starts against a downgraded copy of the Pi's volume (manual).
- [ ] AC28 — Model text reaches Signal only through `check_outbound`, only
  `outbound.py` makes `CheckedText`, and every `Message` in `src/` states
  its origin
  (`tests/test_invariants.py::test_model_text_reaches_signal_only_through_check_outbound`,
  `tests/test_invariants.py::test_only_outbound_makes_checked_text`,
  `tests/test_invariants.py::test_every_message_states_its_origin`).
- [ ] AC29 — `make live` sends a history with gateway records and fenced
  assistant rows to real Fireworks, and the request is accepted
  (`tests/test_live.py::test_gateway_records_are_accepted`).
- [ ] AC30 — The offline replay over a copy of the Pi's database refuses all
  four echo rows and both model-written previews, and no row the owner marks
  legitimate, before each deploy (manual, `python -m evals.replay`).
- [ ] AC31 — Each deploy's manual Pi checks in Rollout pass, and a week of
  logs shows no `outbound.refused` on a message the owner wanted (manual).
- [ ] AC32 — The architecture, pi-deploy, and release notes describe this
  behaviour, and spec 015's superseded points are marked
  (`tests/test_invariants.py::test_relative_doc_links_resolve`, manual read).

Coverage of the findings this spec owns: A-1 (AC1, AC13), A-3 (AC2), A-4
(AC17), B-1 (AC25, enforced at Rollout step 15), B-9 (AC23), B-10 (AC9), C-5 and C-6
(AC14), C-13 (AC15), D-2 (AC20, AC21), F-2 (AC15), X1-3 (AC6, AC22), X1-4
(AC7, narrowed for counts), X1-5 (AC8), X1-7 (AC3), X2-1 (AC11, AC13), X2-2
(AC12), X2-4 (AC24), X2-5 (AC18), X2-6 (AC4, AC11, AC13), X3-2 (AC10), X4-1
(AC19), X4-4 (AC16), and the Oct 29 impersonated cancel (AC5).

## Security review

- **No tool is added or reclassified.** `mail_inbox`'s description changes,
  but its metadata does not. The staged status result now carries the
  scope's taint. That is a label on a broker-made result, not a
  reclassification.
- **Under taint.** Gating is unchanged. `Scope.from_history` still reads every
  stored row, including legacy rows left out of the request, so taint ages
  out exactly as before. The migration does not touch `untrusted`. Untrusted
  text is fenced in more places than before: untrusted assistant rows,
  untrusted server records, a tainted job's prompt, and a staged result's
  copied arguments. A tainted preview still shows the taint note, and a yes
  still runs with the proposal's taint.
- **Wider confirmation.** More words confirm. A decision still applies only
  to the newest stored preview, and any other message withdraws it, so the
  net effect is that fewer messages can run a stored call than before. Today
  any exact "yes" within 10 minutes runs whatever is stored.
- **Report-only jobs.** The surface is unchanged: no sink, no staging tool,
  no `job_*`. A run can no longer read the conversation at all, trusted or
  untrusted, so neither an injected line nor a stale request can steer it.
  It reads its own run facts (tool names, counts, statuses, all
  server-written) and its prompt. A refused report is never texted.
- **The other person's assistant** learns nothing new. `job_runs` queries
  filter by agent. A relay record in her history is a fenced gateway record,
  as before, and the re-show it can cause on her side tells his assistant
  nothing.
- **Logs.** Refused text is never logged. Only the check, kind, length, and a
  hash prefix are. Repair notes are server text and are never persisted.
- **B-1** makes the no-yes path narrower, never wider.

## Open questions

None are open. The owner answered all eight on 2026-10-30.

1. **Which words confirm and which discard?** Answer: confirm is yes, y,
   yeah, yep, yup, ok, okay, sure, confirm, do it, go ahead, send it, yes
   please, and 👍 sent as text. Discard is no, n, nope, discard, and
   cancel. No reaction support in this spec. Reflected in Design § What a
   yes is bound to (near-yes words), Out of scope (reactions), and AC22.
2. **Should any other message withdraw a live proposal?** Answer: yes. Any
   other message withdraws it, and the model must propose again. Reflected
   in Design § What a yes is bound to (withdrawal) and AC20.
3. **Should any job read the conversation?** Answer: no. A job sees only its
   prompt and its own earlier run facts. Reflected in Design § The job-run
   contract (what a run is shown), Risks (a job that drew on conversation
   context), and AC12.
4. **B-1: should a reminder wait for a yes when its saved sentence is not
   mostly the person's own words?** Answer: yes, accepted as
   [decision 0003](../decisions/0003-reworded-reminders-wait-for-yes.md),
   which extends decision 0001. It ships log-only until the decision and
   the PRD edit land, and the PRD § Jobs sentence is amended in the same
   commit that turns enforcement on. Reflected in Design § Smaller fixes
   (B-1), § Docs that change with the code, Rollout steps 14 and 15, and
   AC25.
5. **Are the caps right?** Answer: yes. 1,900 characters for a job report
   and 3,800 for a reply. Reflected in Design § One outbound check (the
   `length` check), § Config keys, AC4, and the deploy A manual check.
6. **Should every job message end with the "Looked at: mail (3 found)"
   line?** Answer: yes. Reflected in Design § The job-run contract (the
   status table and `format_looks`), AC15, and the deploy A manual check.
7. **Old echo rows: leave them out of requests or delete them?** Answer:
   leave them in storage and exclude them from what the model is sent,
   which is reversible. Reflected in Design § Server text is stored as
   records (the `legacy` rules), § Schema 16 (assistant rows without tool
   calls stay `legacy`), Risks (the migration misclassifies rows), and AC11.
8. **Should a job run see its earlier reports' text?** Answer: no. It sees
   only times, tools, counts, and statuses of earlier runs. Reflected in
   Design § The job-run contract (what a run is shown, and `job_runs`
   storing no report text) and AC12.

PRD conflict: question 4 (B-1) changed PRD behaviour, and decision 0003
resolves it. `prd.md` is not edited by this spec's first commits. Its § Jobs
sentence changes only in Rollout step 15, with enforcement. The rest of the
spec matches the PRD and makes the code match `architecture.md`'s existing
"the next inbound message either confirms it or does not".
