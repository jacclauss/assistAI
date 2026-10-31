# 018 — A durable outbox for every external write, and a job runner of its own

Status: ready

Implements: [prd.md § Jobs](../prd.md) ("Jobs survive a restart", "A job
failure ... messages the owner. No silent skip"),
[prd.md § Relay](../prd.md) (step 3, "A send that already reached the phone
is not repeated"), [prd.md § Memory](../prd.md), and
[architecture.md § Staged actions](../architecture.md) and
[§ Persistence](../architecture.md).
Decision: [0004](../decisions/0004-relays-after-an-outage.md).

Depends on: spec 016 (wall-clock schedules, schema 15) and spec 017 (job-run
contract and the checked outbound path, schema 16 if it needs one). This
spec starts from whatever schema 017 leaves and uses versions 17 to 21.

Audit: option C in the October 30 architecture audit, cause 2.

## Outcome

Every write that leaves the conversation is a row first. A Signal message,
a relay, a calendar add, a Gmail draft or filing, a job run, and an update
notice each have a stored record of which steps finished. A restart, a
deploy, or a cancel in the middle of one of them finishes it, or records
that it stopped and tells the person. It never repeats a step that already
landed, and it never drops one silently. Afterwards:

- A long message that fails partway sends only the parts that did not
  arrive. A proposal cannot be confirmed until its whole preview has
  arrived.
- A reply, preview, summary, or report that was saved is delivered, even
  when the process stopped before sending it. A summary is sent only after
  it is saved.
- A job report or a relay that reached the phone is not sent again after a
  restart.
- A reminder saved in a turn that never finished does not fire, and the
  person is told that turn did not finish.
- A yes that a restart interrupts finishes the calls it had not run, or
  says exactly which ran and which did not.
- A failed "no" does not leave a proposal that history calls discarded.
- A confirmed call that failed is not reported as run, and can be confirmed
  again on its own.
- A repeated or edited inbound message does not run a second turn.
- A lost response from iCloud or Gmail does not write a second event or
  draft when retried.
- A scheduled relay that cannot be sent tells the sender.
- A cancelled relay or report is not sent by a stale due list.
- One person's slow turn does not hold up the other person's reminders.

## Out of scope

- How a job's schedule is stored and its next run computed, the time
  arguments of `job_create`, `relay`, and `job_reschedule`, the local-time
  formatter, and DST handling. Spec 016 owns them.
- The structured result of a job run, how job runs and server-written
  messages look in history, what a job run is shown instead of
  `trusted_context`, the one check every outbound model text passes, the
  report length cap, and what a yes is bound to. Spec 017 owns them.
- D-1, D-7, D-11, G-1, G-2, G-3, G-5, H-4, H-5, A-5, F-5, F-7, X4-6, E-4,
  and C-14. Those are separate bug fixes.
- Attachment bytes on a relay. Gmail scope. Encryption at rest.
- A second daemon. The job runner becomes its own asyncio task in the same
  process. The architecture's "in-process asyncio scheduler" decision
  stands.

## Design

The design ships in six stages. Stage 0 changes no schema. Stages 1 to 5
each add one schema version, ship in one deploy, and roll back on their
own. Each stage works if the later ones never ship.

| Stage | Schema | What becomes durable | Findings closed |
| --- | --- | --- | --- |
| 0 | none | Test doubles, crash harness, stop grace, backup | none (enables the rest) |
| 1 | 17 | Conversation sends and update notices | D-8, E-5, X5-4, G-4, C-9 (conversation half) |
| 2 | 18 | Job reports, notices, and relays | E-3, C-8, C-9 (job half), C-10, C-12, C-4 (send half) |
| 3 | 19 | Proposal states, held reminders, inbound record | E-1, E-2, X5-6, D-10, X1-6, D-3 |
| 4 | 20 | Idempotent provider writes | X5-2, X5-7 |
| 5 | 21 | Job runs, and a runner that takes no conversation lock | C-4, X6-6 |

Severities are the verifiers'. High: C-4, C-9, D-3 (partly), D-8, D-10,
E-1, E-2, E-3, X5-2, X5-6. Medium: C-8, C-10 (partly), C-12, E-5, G-4
(partly), X1-6 (partly), X5-4, X5-7, X6-6. Duplicates folded in: X6-1 into
E-1; X5-3 and X6-2 into E-2; X6-5 into E-3; X6-7 into E-5; D-12 and X6-3
into X5-4; X6-4 into D-10; X5-1 into X1-6; F-4 into X5-2; X5-5, which could
not be verified from code, into D-3.

### Shared pieces

**One writer per effect.** A new module `src/assistai/outbox.py` owns the
`outbox` and `outbox_parts` tables and the `OutboxWorker` task. From stage 2
on it is the only code that calls the Signal transport. The channel, the
job runner, relay, and the update watcher enqueue rows. They do not send.

**Enqueue in the same transaction as the record it belongs to.**
`Store.persist` gains an `outbox` argument. A turn's history, its proposal
change, and its outgoing text commit together or not at all. The same holds
for a job run's advance and its report, and for an update notice and its
saved report. Nothing is sent that the database does not hold, and nothing
the database holds as said is left unsent.

**Per-part status.** Text is split into parts of at most
`MAX_SEND_CHARS` (1,900) when it is enqueued, not when it is sent, so a
retry sends the same parts. Each part is `pending`, `sending`, `sent`,
`uncertain`, or `failed`. The worker writes `sending` before the POST and
`sent` with the signal-cli timestamp after it. `SignalTransport` gains
`send_part(recipient, text) -> int | None`, which is today's
`SignalClient._send_part` returning the timestamp from the `/v2/send`
response. `SignalClient.send` and `_chunks` go away in stage 2.

**Uncertain parts.** Signal has no idempotency key. A part that was
`sending` when the process died may or may not have arrived. At start, the
recovery step marks every `sending` part `uncertain`. An uncertain part is
sent once more, preceded by the fixed line "Sent again after a restart. You
may already have this part." This holds for every outbound message,
relays included. On a relay the line goes in front of the stored bytes,
which are unchanged ([decision 0004](../decisions/0004-relays-after-an-outage.md)).
The window is one HTTP POST plus the SQLite update after it. The shutdown
drain below lets an in-flight POST finish, so only a SIGKILL or a power cut
leaves one.

**Order and retries.** The worker sends one row at a time per recipient,
oldest first, so a reply and a report to the same phone arrive in the order
they were saved. Different recipients do not wait on each other. A failed
part waits 30 seconds, doubling to 600, as `JobRunner._defer` does today.
Job-origin rows are metered by `signal_job_messages_per_hour`. Conversation
rows are not, as today. `last_error` holds a fixed code (`http_429`,
`http_4xx`, `http_5xx`, `unreachable`, `timeout`), never a response body,
because a signal-cli 400 body can carry the message text.

**Taint travels with the row.** Each outbox row stores `untrusted`. Any
history it writes later carries that label. Recovery and failure notices
are server text with no outside text in them, and are stored trusted.

**Recovery at start.** A new module `src/assistai/recovery.py` runs once
after the store opens and before the receive loop, the job runner, and the
outbox worker start. Each stage adds its own step to it. It only reads
rows, writes states, and enqueues notices. It never calls a provider or the
model.

**Shutdown as a recorded outcome.** `Gateway._stop_signal` stops in this
order: stop the receive loop and new job dispatch; give in-flight turns
`shutdown_grace_seconds` (5) and then cancel them; let the outbox worker
finish the part it is sending and any rows it can within
`shutdown_drain_seconds` (10); then close. Cancellation is not a failure
path that has to run cleanup. Every step that matters is already a row, so
a cancel or a kill leaves rows that recovery understands. The compose file
gives the gateway `stop_grace_period: 30s`, and `docs/pi-deploy.md` deploys
with `up -d --no-build --timeout 30`, so a drain fits inside Docker's
grace. If Docker still kills at 10 seconds, nothing is lost; recovery has
more to do.

**Backups and downgrade.** Before `Store._init_schema` applies any
migration, it writes `assistai.sqlite.pre-v{N}.bak` with
`sqlite3.Connection.backup`, mode 600, keeping the last three. Stdlib only;
no new dependency. The `python -m assistai store` command group that 016
adds (`jobs` and `downgrade --to N`) gains two subcommands: `backup`, and
`check` (open a file, migrate it, print row counts per table and what each
migration moved). Each stage ships the reverse of its own migration in
`downgrade`. It refuses while a row of that
stage is unfinished (an unsent part, a committing proposal, a running job
run) unless `--force` is given, and prints what `--force` would drop. An
older image still refuses a newer file, which is the safe default
(`store.py:661`). The downgrade is what makes the file openable by the
previous image again.

### Stage 0: harness and grace, no schema change

- `tests/signal_fakes.py` gains `PartedSignal`. It splits at 1,900
  characters, records each part, returns a timestamp per part, can fail
  after part n, can hang on part n until a test releases or cancels it,
  and can deliver an `editMessage` and the same frame twice. `FakeSignal`
  stays for tests that do not care.
- `tests/crash_fakes.py` gives named checkpoints in the store and the
  transport. At a checkpoint a test can cancel the task, or simulate a
  kill: close the store connection so later writes fail, and raise a
  `KilledProcess` that derives from `BaseException`, so `except Exception`
  handlers do not run. The test then opens a fresh `Store` on the same file
  and a fresh channel, runner, and worker, and runs recovery.
- Every test in "Tests that fail first" below lands here, marked
  `xfail(strict=True)` with the finding id in the reason. `make check`
  then proves each one fails on today's code. Each later stage removes the
  marker from the tests it fixes.
- `docker/docker-compose.yml` sets `stop_grace_period: 30s` on `gateway`.
  `docs/pi-deploy.md` adds `--timeout 30` to the deploy command and a
  rollback section.
- `Store._init_schema` writes the backup before migrating.
  `python -m assistai store backup` and `python -m assistai store check`
  exist.

### Stage 1: conversation sends (schema 17)

Tables:

- `outbox`: `id` TEXT primary key, `kind` (`reply`, `preview`, `summary`,
  `notice`, `system`, `update_notice`), `agent` (null for `system`, the
  pairing and unbound replies), `recipient`, `untrusted`, `created_at`,
  `not_before`, `status` (`pending`, `sending`, `sent`, `cancelled`,
  `withdrawn`), `attempts`, `next_attempt_at`, `last_error`, `source_kind`,
  `source_id`.
- `outbox_parts`: `outbox_id`, `idx`, `text`, `status`, `sent_at`,
  `signal_ts`, primary key `(outbox_id, idx)`.
- `proposals` gains `preview_outbox_id` and `opened_at`.

Changes:

- `SignalChannel._converse`, `_commit_proposal`, `_reject_proposal`,
  `_handle_approve`, `_handle_unknown`, and the unbound, too-long, and
  too-fast replies stop calling `_safe_send`. They pass an outbox row to
  `Store.persist`, or to `Outbox.enqueue` when there is no history, and wake
  the worker. `_safe_send` is removed.
- A proposal is confirmable only once `opened_at` is set. The worker sets
  it in the same transaction that marks the preview's last part `sent`.
  `Store.take_proposal` treats a row without `opened_at` as not yet
  confirmable. A whole-message yes in that state gets the fixed sentence
  "That proposal has not finished arriving, so nothing ran." The yes does
  not go to the model. The proposal lifetime counts from `opened_at`.
- A preview whose proposal is older than `staging_ttl_seconds` when its
  first part would go out is `withdrawn`. The proposal row is cleared, and a
  notice says "I could not show you that proposal in time, so nothing will
  run. Ask again if you still want it." This covers a preview saved just
  before a long outage.
- `_commit_proposal` no longer sends the summary when its save fails. The
  summary and the yes are saved, then sent. If the save fails, the commit is
  logged `staging.summary_unsaved` and retried by recovery at the next start
  and by the worker's next pass. The phone never carries a summary that
  history lacks. This also removes the path where `_append_untrusted` or
  `persist_job_report` later reloads over a dirty cache.
- `UpdateWatcher.check` and `_report_confirmed` save the report and enqueue
  its notice in one transaction. A failed save sends nothing, and the next
  check sends it once.
- Recovery marks `sending` parts `uncertain` and applies the rule above. A
  preview with an uncertain part does not open until that part is resolved.

Migration from 16: create the tables and columns. A live `proposals` row
gets `opened_at = created_at`, because its preview was sent by the old code
before the deploy. Downgrade to 16 refuses while any outbox row is unsent.
Otherwise it drops the tables and columns.

### Stage 2: job reports, notices, and relays (schema 18)

Tables:

- `outbox` gains kinds `job_report`, `job_notice` (failure and expiry),
  `reminder`, `relay`, `relay_confirmation`, and a `source_job_id`.
- `outbox_steps`: `outbox_id`, `idx`, `step` (`history`, `relay_record`),
  `payload` (JSON), `status` (`pending`, `done`, `failed`), `attempts`,
  `done_at`. Steps run after every part is `sent`, in order.

Changes:

- `JobRunner._queue` and `_deliver` are replaced by one store method,
  `Store.queue_job_send(job_id, row, *, after)`. In one transaction it
  checks the job is still active, inserts the outbox row and its steps, and
  either advances the job or cancels it. The advance writes the next run
  that spec 016 computes. This spec does not compute it. `jobs.pending_text`,
  `pending_untrusted`, `pending_after`, `pending_delivered`, `relay_sent`,
  and `relay_recorded` stop being read or written.
- The `history` step appends what spec 017 says a job run or notice looks
  like in history. The payload is the message list 017 builds, stored as
  is. The step calls the channel's history appender until stage 5.
- A relay due at its time becomes an outbox row with kind `relay`, parts
  addressed to the other phone, a `relay_record` step for her history, and
  then a `relay_confirmation` row to the sender ("Sent to the other phone"),
  enqueued only when the record step is done. That keeps the PRD rule that
  the sender is told only after her record is stored. `deliver_stored_relay`
  splits into `enqueue_relay` and the record step. An uncertain relay part
  is sent once more after the resend line, in front of the unchanged stored
  bytes. Her record stores the body only, without that line.
- An immediate relay at commit enqueues the same row. The handler waits for
  its parts, up to 30 seconds, before returning. A record that fails is
  retried by its step and never resends the text. If the parts have not
  finished in 30 seconds, the result says the message is queued, and the
  confirmation row tells the sender when it lands.
- The in-memory sets `_relay_sent`, `_relay_recorded`, and `_notified` are
  removed. The rows are the record.
- A relay that cannot be sent tells the sender after its third failed
  attempt: "Your message to {name}, due at {time}, has not gone out yet.
  I am still trying." A relay may go out up to one hour after its time
  (`relay_late_limit_seconds`, 3600). Past that, it is `cancelled`, and the
  sender is told "Your message to {name}, due at {time}, was not delivered."
  The limit counts from the relay's scheduled time, so it also covers a Pi
  that was off at that time. The time uses spec 016's formatter
  ([decision 0004](../decisions/0004-relays-after-an-outage.md)).
- prd.md § Relay is amended in the commit that ships this stage, for the
  resend line on relays and the one-hour limit, as decision 0004 requires.
- Before the first part of a `relay`, `reminder`, or `job_report` row goes
  out, the worker checks `jobs.cancelled_at` for `source_job_id`. A
  cancelled job's unsent row is `cancelled` and logged
  `outbox.cancelled_before_send`. A row already sending finishes. The
  `job_cancel` result says so when one was in flight: "Its last message was
  already going out."

Migration from 17, for live rows:

- An active job with `pending_text` and `pending_delivered = 0` becomes a
  `pending` outbox row with all parts `pending` and a `history` step.
- Any job, active or cancelled, with `pending_delivered = 1` becomes a
  `sent` row with its `history` step `pending`. The phone already has it.
- A cancelled job with `pending_delivered = 0` is dropped and counted in
  `store.migrated` as `dropped_cancelled`. Cancel wins, as in
  `Store.queue_job_report` today.
- A relay with `relay_sent = 1` and `relay_recorded = 0` becomes a `sent`
  row with its `relay_record` step `pending`.
- The migration then clears the old columns. It does not touch any column
  spec 016 owns.

The Pi tonight has one active verbatim reminder ("mgmplus cancel
reminder", first run December 10) and cancelled rows kept for audit. The
migration test uses that shape.

Downgrade to 17 writes each job's unfinished outbox row back into the old
columns. A row with every part sent sets `pending_delivered = 1`. A row with
some parts sent sets it to 0, so the old runner sends the whole text again.
`downgrade` prints how many rows that applies to.

### Stage 3: proposal states, held reminders, inbound record (schema 19)

Tables:

- `staged_proposals`: `id`, `agent`, `calls` (JSON), `tainted`, `state`,
  `created_at`, `opened_at`, `expires_at`, `decided_at`,
  `preview_outbox_id`, `retry_of`, plus every column spec 017 adds to
  `proposals` for what a yes is bound to, copied unchanged. States are
  `staged` (saved, preview not delivered), `open`, `committing`,
  `committed`, `partly_committed`, `discarded`, `expired`, `withdrawn`,
  `interrupted`. One `staged` or `open` proposal per agent, enforced by a
  partial unique index. Rows stay for audit and are pruned after 30 days.
- `proposal_calls`: `proposal_id`, `idx`, `name`, `arguments`,
  `effect_key`, `status` (`pending`, `running`, `done`, `failed`,
  `uncertain`, `abandoned`), `attempts`, `result`, `finished_at`.
- `jobs` gains `held_by`: the id of the inbound turn or proposal that saved
  the row and has not finished. `Store.due_jobs` and `list_jobs` skip held
  rows.
- `inbound`: `sender`, `sent_ts`, `device`, `received_at`, `agent`,
  `kind` (`message`, `edit`), `status` (`handling`, `answered`,
  `interrupted`, `duplicate`, `edit_refused`). Primary key
  `(sender, sent_ts, device)`. No message text. Pruned after 7 days.

Proposals:

- Staging writes a `staged_proposals` row and its `proposal_calls` rows,
  each with a fresh `effect_key`. The preview's last part `sent` moves it to
  `open`. A newer proposal moves an older `staged` or `open` one to
  `withdrawn`.
- Spec 017 decides which inbound message is a yes for which proposal. This
  spec only requires that its check runs against an `open` proposal.
- A yes moves the proposal to `committing` in one transaction with the
  inbound row. Each call is marked `running` before its handler runs and
  `done`, `failed`, or `uncertain` after it. A handler raises
  `OutcomeUnknown` from a new `src/assistai/effects.py` when a provider
  timed out after the request may have been stored. `BoundSurface.commit`
  binds the call's `effect_key` in a context variable next to `using_agent`.
- After the last call, one transaction appends the yes and the summary to
  history, enqueues the summary, arms the reminders this commit saved, moves
  the proposal to `committed` or `partly_committed`, and marks the inbound
  row `answered`.
- `staging.format_executed` lists calls by status. Done calls use their
  `say` line or result. Failed calls go under "Did not run:" with the
  error. Uncertain calls go under "Not sure this ran:" with what to check
  ("look at the calendar for Friday"). The heading "Ran N actions" covers
  only done calls.
- When any call failed, the server stores a new proposal holding only the
  failed calls, with `retry_of` set, and appends its preview to the
  summary: "Reply yes to try these again." It is built from the stored
  calls, like every preview, and opens when the summary has arrived. Done
  and uncertain calls are never in it.
- A "no" moves the proposal to `discarded` and appends the "no" and
  "Discarded." in one transaction. On failure the cached history is rolled
  back to its baseline, the proposal stays `open`, and the reply is "I could
  not save that. The proposal is still open; reply no again." This replaces
  today's append-before-save in `_reject_proposal`.

Held reminders:

- `job_create` and a future relay write `held_by` from a context variable
  the channel sets for the turn or the commit. The turn's
  `Store.persist` clears `held_by` for those ids in the same transaction
  that saves the reply. A turn that fails, is cancelled, or is killed
  leaves them held. A held row cannot fire.
- `_cancel_turn_jobs` still cancels held rows on a failed turn. If that
  cancel fails, the row stays held, so the person's "not saved" is true.
  Recovery cancels it at the next start.
- `holds_job` and `_open_turns` are removed. The column replaces them.

Inbound:

- `SignalChannel.handle` inserts the inbound row after the sender resolves
  and policy allows it, and before any turn. `parse_inbound` keeps the
  envelope `timestamp`, `sourceDevice`, and, for an `editMessage`, its
  `targetSentTimestamp` as `edit_of`. `_data_message` no longer turns an
  edit into a fresh message.
- A row that already exists is a duplicate. It is logged
  `signal.inbound_duplicate` and gets no turn and no reply.
- An edit to a message already answered gets the fixed reply "I already
  answered the first version. Send the change as a new message." and
  nothing else: no model turn, no tool call. The per-sender lock means an
  edit is handled only after its original's turn ends, so the original is
  `answered` or `interrupted` by then. An edit whose original is not in
  `inbound` (never received) or is `interrupted` was never answered, so it
  runs as a new message.
- Both checks ship behind `ASSISTAI_INBOUND_DEDUPE=log`. In `log` mode they
  record and log `would_drop=true` and run the turn as today. The default
  flips to `enforce` in a later commit, after a week of Pi logs.

Recovery adds:

- An inbound row still `handling` becomes `interrupted`. Its held reminders
  are cancelled. The person gets "I restarted while answering your
  {time} message. Nothing from it was saved. Send it again if you still need
  it." The time uses spec 016's formatter. No excerpt of their text is
  stored or sent.
- A `committing` proposal whose yes came less than ten minutes before the
  gateway starts again (`commit_resume_seconds`, 600) finishes
  automatically. `pending` calls run. `running` calls become `uncertain`
  and, until stage 4, are not run again. The summary starts "After a
  restart:". If the gateway starts again ten minutes or more after the yes,
  it stops: `pending` calls become `abandoned`, and the summary says what
  ran, what did not, and what to check. Nothing more runs without a new
  yes.
- A `staged` proposal whose preview row was withdrawn becomes `withdrawn`.

Migration from 18: copy a live `proposals` row into `staged_proposals` as
`open`, with its expiry and every 017 column. Leave `proposals` empty and
unused. Add `held_by` as null, so every existing job stays armed; that is
how it behaved before. Downgrade to 18 copies an `open` proposal back into
`proposals`, refuses while one is `committing`, and drops the rest.

### Stage 4: idempotent provider writes (schema 20)

- `applied_effects`: `key` primary key, `kind`, `result`, `applied_at`.
  A write that is local to SQLite records its key in the same transaction
  as the write.
- `calendar_add`: `CalendarClient._put_new` takes the UID from the effect
  key instead of `uuid.uuid4()`. A PUT with `If-None-Match: *` that gets
  412 on that URL means this event is already stored. The client confirms
  with a GET of the same URL and returns success. It no longer switches to
  a fresh UUID on 409 or 412. A timeout raises `OutcomeUnknown`.
- `mail_draft`: `_rfc822` sets `Message-ID: <{key}@assistai.invalid>`.
  Before a retry POST, `GmailClient.draft` asks `GET /drafts` with
  `q=rfc822msgid:{key}@assistai.invalid`. A match is success. This is a read
  under the existing `gmail.modify` grant, and `_refuse_path` is unchanged.
- `mail_file`: `batchModify` is a label change and safe to repeat. A
  timeout raises `OutcomeUnknown`. A retry checks the preview again in
  `GmailClient.apply` and repeats the change. If Gmail stays unreachable, the
  summary says the outcome is unknown, not that it failed.
- `shared_change`: `Store.apply_shared` takes the key. A second apply with a
  recorded key returns the stored result instead of "already on".
- `job_create`, `job_cancel`, `job_reschedule`: the job id comes from the
  key, and the result is recorded under it. A second run returns the first
  result. The reschedule records 016's computed next run, so a retry does
  not compute it again.
- `relay`: already keyed by its outbox row in stage 2.
- Recovery and the commit loop now retry `uncertain` calls once with the
  same key, then report.

Migration from 19 creates `applied_effects`. Downgrade drops it.

### Stage 5: a job runner of its own (schema 21)

Tables:

- `job_runs`: `id`, `job_id`, `slot` (the instant spec 016 scheduled),
  `status` (`claimed`, `running`, `reported`, `done`, `skipped`,
  `interrupted`), `attempt`, `claimed_at`, `finished_at`, `outcome`
  (spec 017's structured result, stored as 017 defines it), `outbox_id`.
  Unique `(job_id, slot)`.
- `history_meta`: `agent` primary key, `rev` integer.

Changes:

- `JobRunner._tick` dispatches each due job as its own task, bounded by
  `job_concurrency` (default 2), and returns. It no longer awaits one fire
  before the next.
- Each fire re-reads its row by id, claims `(job_id, slot)`, and does
  nothing if the claim exists or the job is cancelled or held. A verbatim
  reminder or a relay is claimed, enqueued, and advanced in one
  transaction, with no model. A digest is claimed, then run, then its
  outcome, report row, and advance commit in one transaction with a
  `cancelled_at IS NULL` guard.
- `run_job` moves from `SignalChannel` into `src/assistai/scheduler.py` as
  `JobModel`. It reads whatever context spec 017 defines straight from
  SQLite. It takes no peer lock, no `_slots` permit, and no history lock.
  `JobChannel` shrinks to nothing. The runner depends on the store, the
  outbox, the broker, and the Fireworks client.
- History gets a revision. Every write bumps `history_meta.rev`.
  `Store.append_history` inserts after the last row and bumps it, with no
  channel lock. That is how the `history` and `relay_record` steps write.
  `Store.persist` takes the revision the channel loaded. If it moved, it
  raises `HistoryConflict`. The channel reloads, puts its turn's new
  messages on top, and saves again, up to three times. A turn's history
  lock stays inside the channel and is never taken by the runner or the
  worker.
- Recovery marks a `running` run `interrupted`. That slot stays due and
  runs once more, because its report never went out. A slot that reached
  `reported` does not run again.

Migration from 20 creates both tables. `history_meta` starts at 0 for each
agent with history. Downgrade drops them, after refusing while a run is
`running`.

### Contracts with 016 and 017

This spec must leave these unchanged:

- Spec 016's schedule columns and next-run computation. This spec never
  writes `next_run_at` or any schedule field except through 016's store
  method. The advance inside `Store.queue_job_send` and the stage 5 run
  transaction asks 016 for the next run after this slot. `job_runs.slot` is
  the instant 016 scheduled. Migrations copy 016's columns untouched. Every
  time in a notice uses 016's formatter. The C-1 tests 016 changes are
  016's. This spec's job tests take their expected next run from 016's
  function, not from a literal.
- Spec 017's structured job result. It is stored in `job_runs.outcome` and
  never re-derived. The runner decides what to send from it as 017 says.
- Spec 017's outbound check. Model text reaches `Outbox.enqueue` only
  after it. The enqueue API takes 017's checked type for model text and a
  separate type for server text. The worker never alters text. It only
  splits it into parts after 017's length cap.
- Spec 017's history representation for job runs, previews, summaries, and
  notices. The `history` step writes the messages 017 builds, unchanged.
  Recovery and failure notices use 017's notice form.
- Spec 017's yes binding. It runs against `open` proposals only. The stage
  3 migration copies 017's proposal columns.

## Risks and mitigations

**It touches the commit path, the scheduler, relay, the Signal client, and
the store at once.** That is the risk the audit named. Mitigation: six
stages, each one deploy and one schema version, each with its own downgrade.
Stage 1 touches only the channel's sends and the update watcher. Stage 2
touches only the runner's sends and relay. Stage 3 touches staging and
inbound. Stage 4 touches only provider clients. Stage 5 touches only the
runner and history writes. No stage ships with a later stage's code paths
enabled.

**Tests that fail first.** These land in stage 0 as strict xfails and come
off as their stage fixes them. They come from report section 6: the Signal
double (missing test 3) and the crash matrix (missing test 4).

- `tests/test_channel.py::test_a_preview_that_did_not_fully_arrive_cannot_be_confirmed` (D-8)
- `tests/test_channel.py::test_a_reply_cancelled_before_send_goes_out_after_restart` (E-5)
- `tests/test_channel.py::test_a_summary_is_sent_only_after_it_is_stored` (X5-4)
- `tests/test_updates.py::test_a_notice_is_saved_before_it_is_sent` (G-4)
- `tests/test_jobs.py::test_a_report_part_failure_does_not_resend_earlier_parts` (C-9)
- `tests/test_jobs.py::test_a_report_sent_before_a_kill_is_not_sent_again_after_restart` (E-3)
- `tests/test_jobs.py::test_a_failed_advance_does_not_append_the_report_twice` (C-8)
- `tests/test_relay.py::test_an_immediate_relay_record_is_retried_without_resending` (C-10)
- `tests/test_jobs.py::test_a_relay_that_cannot_send_tells_the_sender` (C-12)
- `tests/test_jobs.py::test_a_relay_cancelled_mid_tick_is_not_sent` (C-4)
- `tests/test_channel.py::test_a_reminder_saved_in_a_cancelled_turn_never_fires` (E-1)
- `tests/test_channel.py::test_a_failed_rollback_cancel_leaves_the_reminder_held` (X5-6)
- `tests/test_recovery.py::test_an_interrupted_yes_finishes_after_restart` (E-2)
- `tests/test_channel.py::test_a_failed_no_leaves_history_and_proposal_consistent` (D-10)
- `tests/test_staging.py::test_a_failed_call_is_not_reported_as_ran` (X1-6)
- `tests/test_channel.py::test_a_repeated_frame_runs_one_turn` (D-3)
- `tests/test_channel.py::test_an_edit_does_not_run_a_second_turn` (D-3)
- `tests/test_calendar.py::test_a_lost_put_response_retried_stores_one_event` (X5-2)
- `tests/test_mail.py::test_a_lost_draft_response_is_not_saved_twice` (X5-2)
- `tests/test_mail.py::test_a_lost_file_response_is_retried_and_reported_applied` (X5-7)
- `tests/test_jobs.py::test_a_slow_turn_does_not_delay_another_persons_reminder` (X6-6)

**Pinned tests that change, and why that is intended.**

- `tests/test_jobs.py::test_pending_notify_survives_reopen` treats an
  unmarked outbox as unsent after reopen, which is E-3. In stage 2 the
  pending text lives in `outbox`, and a part that was `sending` follows the
  uncertain rule. Its `next_run_at` assertion is 016's and is left to it.
- `tests/test_jobs.py::test_unpersisted_notify_retries_history_only` keeps
  its property (the history write is retried, the send is not). It asserts
  the `history` step instead of `pending_delivered`.
- `tests/test_jobs.py::test_a_relay_that_fails_before_send_retries` asserts
  a silent retry, which is C-12. It keeps the retry and adds the sender's
  notice after the third attempt.
- `tests/test_relay.py::test_failed_inject_still_delivered` expects
  `RelayError` after the phone has the text, which is C-10. It becomes a
  queued record step and a result that says the message was sent.
- `tests/test_envelopes.py::test_reads_edits` asserts an edit parses as new
  text, which is D-3. It asserts `edit_of` instead.
- `tests/test_jobs.py::test_a_report_whose_delivered_mark_fails_is_not_sent_again`,
  `test_a_relay_whose_sent_mark_fails_is_not_sent_again`, and
  `test_a_relay_record_whose_mark_fails_is_not_written_again` rely on the
  in-memory sets. They fail the step-status write instead, and assert no
  resend in the same process and in a new one.
- `tests/test_channel.py::test_a_reminder_does_not_fire_before_its_turn_ends`
  and `tests/test_jobs.py::test_a_reminder_confirmed_by_yes_waits_for_the_summary`
  keep their assertions. They now pass through `held_by`, not `holds_job`.
- `tests/test_jobs.py::test_a_job_takes_a_turn_slot_before_the_history_lock`
  pins the lock order that causes X6-6. Stage 5 replaces it with
  `test_the_job_runner_does_not_reach_into_the_channel`.
- `tests/test_channel.py::test_one_yes_runs_every_staged_call`,
  `test_yes_executes_the_stored_call_after_a_restart`,
  `test_a_failed_turn_cancels_the_reminder_it_saved`, and
  `tests/test_gateway.py::test_a_hung_channel_is_cancelled_at_shutdown` keep
  their assertions unchanged. They must still pass at every stage.

**Migrating live rows.** The Pi holds one active repeating reminder,
cancelled rows kept for audit, a history window with long stored reports,
and possibly a live proposal. Mitigation: a fixture builder,
`tests/store_fixtures.py::pi_shaped(version)`, creates a file at the
version 017 leaves, using that commit's schema dump, and fills it with
those shapes. It uses placeholder text, never real household text. Each
stage's migration test opens it. Each migration runs in one transaction and
logs `store.migrated` with counts per table and per rule. Before each
deploy, the new image's `python -m assistai store check` runs on the Pi
against a copy of the state file in a second volume, with no network. The
state file never leaves the Pi. The steps are under Rollout. The backup is
written before any migration.

**Rolling back the schema.** Every 0.1.x image refuses a newer file. The
rollback for any stage is: stop the gateway; run `python -m assistai store
downgrade --to N` from the new image with `docker compose run --rm --no-deps
gateway store downgrade --to N`; start the previous image tag. `downgrade`
refuses while that stage has unfinished rows, and the stop drains them
first. The `.bak` file is the fallback. Restoring it loses whatever was
written since the deploy, so it is the last resort. Each stage's downgrade
has a test that opens the downgraded file with the previous version's
`SCHEMA_VERSION` and schema. Before each deploy, the dry run on the Pi also
downgrades the migrated copy and opens it with the previous image's
`store jobs`, in the second volume, with no network.

**Duplicates or gaps from uncertain parts.** A kill inside one POST can
repeat or drop that part. Mitigation: the drain lets the POST finish. A part
POST gets its own 15-second timeout, `signal_part_timeout_seconds`, instead of
the 60-second request timeout, so the drain is bounded. An uncertain part
is sent once more after the resend line, so the failure mode is a labelled
repeat, never a silent gap (decision 0004). Logs `outbox.part_uncertain`.
On a deploy with nobody texting, the expected count is zero.

**Late messages after an outage.** A reply saved before a long outage goes
out late, which beats silence (PRD: "Failures never fail closed with
silence"). A preview that would arrive after its proposal's lifetime is
withdrawn, not opened. A scheduled relay goes out at most one hour late;
after that it is dropped and the sender is told (decision 0004).

**A stuck or crashed worker leaves everything queued.** Mitigation: the
worker loop logs and continues on any exception, as `JobRunner.run` does.
`gateway.heartbeat` gains `outbox_pending` and `oldest_pending_seconds`. A
row pending more than ten minutes logs `outbox.stuck` once.

**Dedupe drops a real message.** Two real messages from one sender and
device with the same millisecond timestamp are not expected, but this is
the one check here that can suppress a real message. There is no model-text
check in this spec, so no replayed-history eval is needed for it. The
log-only week stands in for one. Mitigation: `log` mode first, then
`enforce` after counting `signal.inbound_duplicate` with `would_drop=true`
on the Pi. A drop logs the sender and timestamp, never text.

**Refusing edits changes something the household may rely on.** Mitigation:
log-only first (`signal.edit_refused` with `would_refuse=true`), a release
note, and Jacob tells his wife before the enforce flip. The refusal covers
only edits to messages already answered.

**Restart notices become noise.** Mitigation: a notice goes only to a person
whose turn was actually interrupted. The drain finishes most turns. Watch
`recovery.turn_interrupted` per deploy. If it is above zero on deploys when
nobody was texting, that is a bug, not noise.

**Resuming a yes after a restart surprises someone.** Mitigation: only when
the gateway is back within ten minutes of the yes, and the summary says
"After a restart:". Later than that, it stops and says what ran.

**iCloud or Gmail behave differently from the fakes.** The 412 rule assumes
iCloud honours `If-None-Match: *` on an existing URL. The draft lookup
assumes `drafts.list` accepts `rfc822msgid:`. Mitigation: two live tests,
`tests/test_live.py::test_calendar_put_twice_with_one_uid_is_one_event` and
`tests/test_live.py::test_a_draft_is_found_by_its_message_id`, run with
`make live` before stage 4 ships. If either fails, that provider keeps
today's behaviour and its uncertain calls are reported, not retried. The
event and the draft the live tests create are removed by hand.

**History appends race a turn's save.** Mitigation: the revision check, and
a test that interleaves a report append with a turn at every await, using
the crash harness checkpoints.

**Taint lost on the way through the outbox.** Mitigation:
`tests/test_outbox.py::test_an_outbox_record_keeps_its_untrusted_label`
and the existing taint tests run unchanged at every stage.

**Secrets reaching the database or logs through error text.** Mitigation:
`last_error` and step errors hold fixed codes only.
`tests/test_outbox.py::test_outbox_errors_hold_no_response_body`. Backups
hold the Gmail refresh tokens the live file holds, so they are mode 600 in
the same state volume, which is the documented trust boundary.

**More model calls at once.** Job runs get their own `job_concurrency`
permits instead of sharing `max_concurrent_turns`. Fireworks concurrency can
reach 6 instead of 4. That is acceptable on the Pi, which only waits on the
network.

**What to watch on the Pi after each deploy.** `store.backup_written`,
`store.migrated`, `outbox.enqueued`, `outbox.delivered`,
`outbox.part_failed`, `outbox.part_uncertain`, `outbox.stuck`,
`outbox.cancelled_before_send`, `outbox.withdrawn`, `staging.opened`,
`staging.call_done`, `staging.call_failed`, `staging.call_uncertain`,
`staging.summary_unsaved`, `recovery.turn_interrupted`,
`recovery.proposal_resumed`, `recovery.proposal_abandoned`,
`recovery.reminder_released`, `signal.inbound_duplicate`,
`signal.edit_refused`, `jobs.run_claimed`, `jobs.run_reported`,
`shutdown.drain_started`, `shutdown.turn_cancelled`, `shutdown.drained`.
`signal.send_failed` and `store.save_failed_after_commit` should stop
appearing after stage 2.

**Telling the household.** Each stage adds a release-notes entry in plain
words. Jacob tells his wife before stage 3 enforces: an edited message gets
a "send it as a new message" reply, and after a restart the bot may text
that it did not finish her last message. Stage 2: a later message that
cannot go out now says so, a message more than an hour late is dropped,
and a message cut off by a restart may arrive again under "Sent again after
a restart". No bot broadcast; that would be a new kind of unprompted
message.

## Rollout

Each deploy is one stage. On the Mac, `make check` passes. On the Pi, the
state file never leaves the Pi, and the dry run touches only a copy:

1. Save the running image as `docs/pi-deploy.md` says, and tag it
   `assistai/gateway:previous`. Load the new tar.
2. `docker compose --env-file .env -f docker/docker-compose.yml run --rm
   --no-deps gateway store backup` writes a consistent copy of the live file
   while the gateway runs.
3. Create a second volume, `assistai_gateway-state-dryrun`, and copy only
   that backup into it as `assistai.sqlite`, with a throwaway `alpine`
   container.
4. Against that volume, with `--network none` so nothing can reach Signal or
   a provider: the new image's `store check` must migrate it and print the
   expected counts; the new image's `store downgrade --to N` must succeed;
   `assistai/gateway:previous` running `store jobs` must open the result and
   list the active reminder.
5. Remove the dry-run volume.
6. Deploy: `docker compose --env-file .env -f docker/docker-compose.yml up
   -d --no-build --timeout 30`.

Stages:

1. **Stage 0.** Commits: the `PartedSignal` double; the crash harness; the
   strict xfail tests; `stop_grace_period` and the deploy timeout;
   backup-before-migrate and `python -m assistai store backup` and `check`.
   Pi check:
   `docker inspect` on the gateway container shows `StopTimeout` 30, and
   `store backup` writes a mode-600 file in the state volume.
2. **Stage 1, schema 17.** Commits: the outbox tables and worker; channel
   sends through the outbox; proposals open on delivery; summary only after
   save; update notice saved with its send; recovery for uncertain parts;
   `python -m assistai store downgrade --to 16`; docs. Pi check: ask a question whose answer is
   over 1,900 characters and see `outbox.delivered parts=2` or more; stage a
   relay to yourself, check `staging.opened` follows the preview's last
   part, and say no.
3. **Stage 2, schema 18.** Commits: job sends through the outbox; migration
   of the old outbox columns; relay record as a step; sender notices; cancel
   check before first part; removal of the in-memory sets; the "only the
   outbox sends" invariant; docs. Pi check: `store.migrated` shows the
   reminder row untouched and zero `dropped_cancelled` unless expected;
   create a two-minute test reminder and see `outbox.delivered`; create a
   two-minute test relay to the other phone, cancel it, and see
   `outbox.cancelled_before_send`.
4. **Stage 3, schema 19, dedupe in `log` mode.** Commits: proposal states
   and per-call status; held reminders; inbound record; edit handling;
   failed-no rollback; status-aware summary and retry proposal; recovery for
   turns and commits; docs. Pi check: text "remind me in 5 minutes to test",
   deploy the same image again with `--force-recreate` while the next
   message is mid-turn, and see `recovery.turn_interrupted` and the notice;
   see the 5-minute reminder still fire.
5. **Stage 3b.** One commit flips `ASSISTAI_INBOUND_DEDUPE` to `enforce`,
   after a week of `would_drop` and `would_refuse` counts are read on the
   Pi. Pi check: edit a sent message and get the fixed reply.
6. **Stage 4, schema 20.** Commits: `applied_effects`; keyed calendar PUT;
   draft Message-ID and lookup; filing retry; keyed list and job writes;
   uncertain retry on resume; docs. Before it ships, `make live` with the two
   provider tests. Pi check: confirm one calendar add and one draft, and see
   one of each.
7. **Stage 5, schema 21.** Commits: `job_runs`; per-job tasks; `JobModel`
   out of the channel; `history_meta` and `append_history`; recovery for
   runs; docs. Pi check: while one phone is mid-turn, a reminder on the
   other phone fires on time; `jobs.run_claimed` and `jobs.run_reported`
   appear once per slot.

The rollback image for each stage is the previous stage's tag, after
`python -m assistai store downgrade --to N`.

## Acceptance criteria

### Stage 0

- [ ] AC0.1 — The Signal double splits at 1,900 characters, returns a timestamp per part, fails or hangs after part n, and delivers an edit and a repeated frame (`tests/test_signal_fakes.py::test_parted_signal_behaves_like_the_client`)
- [ ] AC0.2 — The crash harness cancels or kills at a named point and reopens the same file in a new stack (`tests/test_crash_fakes.py::test_a_kill_skips_later_writes_and_reopens`)
- [ ] AC0.3 — Every test under "Tests that fail first" exists, is a strict xfail, and names its finding (`make check`, manual read)
- [ ] AC0.4 — A migration writes a mode-600 backup first (`tests/test_store.py::test_a_migration_writes_a_private_backup_first`)
- [ ] AC0.5 — The gateway gets 30 seconds to stop (manual: `docker inspect` on the Pi shows `StopTimeout` 30)
- [ ] AC0.6 — Before every deploy of stages 1 to 5, the dry run under Rollout passes on the Pi against a copy in a second volume with no network, and the dry-run volume is removed afterwards (manual)

### Stage 1 (schema 17)

- [ ] AC1.1 — A reply, preview, summary, or notice is queued in the same transaction as its history, and the channel never calls the transport (`tests/test_outbox.py::test_a_reply_is_queued_with_its_history`, `tests/test_invariants.py::test_the_channel_does_not_send_directly`)
- [ ] AC1.2 — A long reply whose second part fails sends only the missing parts on retry. Closes C-9, conversation half (`tests/test_outbox.py::test_a_failed_part_is_retried_without_resending_earlier_parts`)
- [ ] AC1.3 — A preview that did not fully arrive cannot be confirmed, and a yes gets the fixed sentence. Closes D-8 (`tests/test_channel.py::test_a_preview_that_did_not_fully_arrive_cannot_be_confirmed`)
- [ ] AC1.4 — A reply saved and then cancelled before its send goes out after a restart, and a preview too old by then is withdrawn with a notice. Closes E-5 and X6-7 (`tests/test_channel.py::test_a_reply_cancelled_before_send_goes_out_after_restart`, `tests/test_channel.py::test_a_stale_preview_is_withdrawn_not_sent`)
- [ ] AC1.5 — A summary whose save fails is not sent until it is stored, and the next turn sees it. Closes X5-4, D-12, and X6-3 (`tests/test_channel.py::test_a_summary_is_sent_only_after_it_is_stored`)
- [ ] AC1.6 — An update notice is saved and queued together; a failed save sends nothing, and a later check sends it once. Closes G-4 (`tests/test_updates.py::test_a_notice_is_saved_before_it_is_sent`)
- [ ] AC1.7 — A part in flight at a kill is uncertain at the next start and is sent once more after "Sent again after a restart. You may already have this part.", and only once (`tests/test_outbox.py::test_a_part_in_flight_at_a_kill_is_uncertain_after_restart`, `tests/test_outbox.py::test_an_uncertain_part_is_resent_once_after_the_marker`)
- [ ] AC1.8 — Outbox rows keep their untrusted label, and their errors hold no response body (`tests/test_outbox.py::test_an_outbox_record_keeps_its_untrusted_label`, `tests/test_outbox.py::test_outbox_errors_hold_no_response_body`)
- [ ] AC1.9 — A Pi-shaped version 16 file migrates, a live proposal stays confirmable, and the downgrade opens as version 16 (`tests/test_store.py::test_schema_v16_gains_the_outbox_and_keeps_rows`, `tests/test_store.py::test_downgrade_from_v17_opens_as_v16`)
- [ ] AC1.10 — Architecture, deploy notes, and release notes describe the outbox and the rollback (`tests/test_invariants.py::test_relative_doc_links_resolve`, manual read)

### Stage 2 (schema 18)

- [ ] AC2.1 — A report sent before a kill is not texted again by a new process. Closes E-3 and X6-5 (`tests/test_jobs.py::test_a_report_sent_before_a_kill_is_not_sent_again_after_restart`)
- [ ] AC2.2 — A report whose later part fails resends only that part. Closes C-9, job half (`tests/test_jobs.py::test_a_report_part_failure_does_not_resend_earlier_parts`)
- [ ] AC2.3 — The report and the advance commit together, so a failing advance does not append the report twice. Closes C-8 (`tests/test_jobs.py::test_a_failed_advance_does_not_append_the_report_twice`)
- [ ] AC2.4 — An immediate relay whose record fails is recorded later without a second send, and the sender is told it was sent. Closes C-10 (`tests/test_relay.py::test_an_immediate_relay_record_is_retried_without_resending`)
- [ ] AC2.5 — A relay that cannot be sent tells the sender after three attempts; it may still go out up to one hour after its time, and after that it is dropped and the sender is told. Closes C-12 (`tests/test_jobs.py::test_a_relay_that_cannot_send_tells_the_sender`, `tests/test_jobs.py::test_a_relay_fifty_nine_minutes_late_still_goes_out`, `tests/test_jobs.py::test_a_relay_past_its_late_limit_is_dropped_and_said`)
- [ ] AC2.5a — An uncertain relay part is sent once more with the resend line in front of the unchanged stored bytes, and her record holds the body only (`tests/test_jobs.py::test_an_uncertain_relay_part_is_resent_after_the_marker`)
- [ ] AC2.6 — A relay or report whose job is cancelled before its first part is not sent. Closes the send half of C-4 (`tests/test_jobs.py::test_a_relay_cancelled_mid_tick_is_not_sent`, `tests/test_jobs.py::test_a_queued_report_of_a_cancelled_job_is_not_sent`)
- [ ] AC2.7 — The sender is told a relay went out only after her record is stored (`tests/test_jobs.py::test_a_relay_confirmation_waits_for_the_record`)
- [ ] AC2.8 — Only `outbox.py` calls the Signal transport (`tests/test_invariants.py::test_only_the_outbox_sends_signal`)
- [ ] AC2.9 — Version 17 job outboxes migrate: delivered ones are not resent, cancelled undelivered ones are dropped, 016's columns are untouched, and the downgrade restores the old columns (`tests/test_store.py::test_schema_v17_moves_job_outboxes_into_the_outbox`, `tests/test_store.py::test_downgrade_from_v18_restores_job_outbox_columns`)
- [ ] AC2.10 — Docs and release notes describe job sends, relay notices, the resend line, and the late limit, and prd.md § Relay is amended in this stage's commit as decision 0004 requires (`tests/test_invariants.py::test_relative_doc_links_resolve`, manual read)

### Stage 3 (schema 19)

- [ ] AC3.1 — A reminder saved in a turn that shutdown cancels never fires, and the person is told at the next start. Closes E-1 and X6-1 (`tests/test_channel.py::test_a_reminder_saved_in_a_cancelled_turn_never_fires`, `tests/test_recovery.py::test_an_interrupted_turn_is_reported_after_restart`)
- [ ] AC3.2 — A reminder whose rollback cancel fails stays held, cannot fire, and is cancelled by recovery. Closes X5-6 (`tests/test_channel.py::test_a_failed_rollback_cancel_leaves_the_reminder_held`)
- [ ] AC3.3 — A yes interrupted after its first call finishes the rest once when the gateway is back within ten minutes of the yes; at ten minutes or later it runs nothing more and reports what ran and what did not. Closes E-2, X5-3, and X6-2 (`tests/test_recovery.py::test_an_interrupted_yes_finishes_after_restart`, `tests/test_recovery.py::test_an_old_interrupted_yes_reports_what_ran`)
- [ ] AC3.4 — A "no" whose save fails leaves the proposal open and history unchanged. Closes D-10 and X6-4 (`tests/test_channel.py::test_a_failed_no_leaves_history_and_proposal_consistent`)
- [ ] AC3.5 — A failed confirmed call is listed as not run, and the summary offers a yes that retries only the failed calls. Closes X1-6 and X5-1 (`tests/test_staging.py::test_a_failed_call_is_not_reported_as_ran`, `tests/test_channel.py::test_failed_calls_can_be_confirmed_again_alone`)
- [ ] AC3.6 — The same frame delivered twice runs one turn; in log mode it is logged and still runs. Closes D-3 and X5-5 (`tests/test_channel.py::test_a_repeated_frame_runs_one_turn`, `tests/test_channel.py::test_dedupe_log_mode_only_logs`)
- [ ] AC3.7 — An edit to a message already answered gets the fixed reply and nothing else, and parses with `edit_of`; an edit whose original was never answered runs as a new message. Closes D-3 (`tests/test_channel.py::test_an_edit_does_not_run_a_second_turn`, `tests/test_channel.py::test_an_edit_of_an_unanswered_message_runs_as_new`, `tests/test_envelopes.py::test_reads_edits`)
- [ ] AC3.8 — 017's yes binding is applied only to an open proposal (`tests/test_channel.py::test_a_yes_binds_only_to_an_open_proposal`)
- [ ] AC3.9 — A live version 18 proposal, with 017's fields, migrates as open; existing jobs stay armed; the downgrade restores `proposals` (`tests/test_store.py::test_schema_v18_gains_proposal_states_and_keeps_a_live_proposal`, `tests/test_store.py::test_downgrade_from_v19_restores_the_live_proposal`)
- [ ] AC3.10 — On the Pi, a week of `would_drop` and `would_refuse` counts is read before the enforce flip, and a redeploy during a turn produces one notice (manual)- [ ] AC3.11 — Architecture's staged-actions section describes the states, the open-on-delivery rule, and resume (manual read)

### Stage 4 (schema 20)

- [ ] AC4.1 — A calendar add whose response is lost and is retried stores one event. Closes X5-2 for the calendar (`tests/test_calendar.py::test_a_lost_put_response_retried_stores_one_event`)
- [ ] AC4.2 — A Gmail draft whose response is lost is found by its Message-ID and not saved twice. Closes X5-2 for mail, and F-4 (`tests/test_mail.py::test_a_lost_draft_response_is_not_saved_twice`)
- [ ] AC4.3 — A filing whose response is lost is retried and reported applied, and an unreachable Gmail is reported as unknown, not failed. Closes X5-7 (`tests/test_mail.py::test_a_lost_file_response_is_retried_and_reported_applied`, `tests/test_mail.py::test_an_unreachable_filing_is_reported_unknown`)
- [ ] AC4.4 — A list change and a job write are applied once per key (`tests/test_shared.py::test_a_list_change_is_applied_once_per_key`, `tests/test_jobs.py::test_a_job_create_is_saved_once_per_key`)
- [ ] AC4.5 — An uncertain call is retried with its key on resume (`tests/test_recovery.py::test_an_uncertain_calendar_add_is_retried_with_its_key`)
- [ ] AC4.6 — The draft lookup is a read, and send and delete are still refused (`tests/test_mail.py::test_draft_lookup_is_a_read`, existing scope tests)
- [ ] AC4.7 — iCloud and Gmail behave as the fakes assume (`tests/test_live.py::test_calendar_put_twice_with_one_uid_is_one_event`, `tests/test_live.py::test_a_draft_is_found_by_its_message_id`, run with `make live`)
- [ ] AC4.8 — Migration and downgrade (`tests/test_store.py::test_schema_v19_gains_applied_effects`, `tests/test_store.py::test_downgrade_from_v20_opens_as_v19`)

### Stage 5 (schema 21)

- [ ] AC5.1 — One person's slow turn does not delay the other person's due reminder. Closes X6-6 (`tests/test_jobs.py::test_a_slow_turn_does_not_delay_another_persons_reminder`)
- [ ] AC5.2 — The runner takes no channel lock and does not import `signal.channel` (`tests/test_invariants.py::test_the_job_runner_does_not_reach_into_the_channel`)
- [ ] AC5.3 — A relay cancelled while another job runs in the same tick is not sent. Closes C-4 (`tests/test_jobs.py::test_a_relay_cancelled_during_another_run_is_not_sent`)
- [ ] AC5.4 — A slot runs once; a restart during a run re-runs that slot once, and a reported slot never runs again (`tests/test_jobs.py::test_a_slot_runs_once_across_restarts`)
- [ ] AC5.5 — A report appended during a turn survives that turn's save (`tests/test_channel.py::test_a_report_appended_mid_turn_survives_the_turn_save`)
- [ ] AC5.6 — Migration and downgrade (`tests/test_store.py::test_schema_v20_gains_job_runs_and_history_revisions`, `tests/test_store.py::test_downgrade_from_v21_opens_as_v20`)
- [ ] AC5.7 — Architecture's jobs section describes the runner as its own task, still in-process (manual read)

## Security review

- No tool is added or reclassified. `relay` stays `staging` with sink
  `message:other_peer`. Jobs stay report-only. The outbox worker and
  recovery run no model and call no tool.
- Taint: every outbox row and step carries its `untrusted` label into the
  history it writes. A relay record is untrusted on her side, as today.
  Recovery and failure notices hold no outside text. No new path clears a
  label.
- Staging: a proposal becomes confirmable later than today, only after the
  person has seen all of it. Resuming a yes after a restart runs only calls
  the person confirmed, with the stored arguments, and only within ten
  minutes of that yes. The retry proposal is built from stored failed calls
  and needs its own yes.
- Relays: the resend line is a fixed server line placed outside the stored
  bytes, which are unchanged. It holds no model or outside text, and her
  record stores the body only (decision 0004).
- The migration dry run runs on the Pi with `--network none` against a copy
  in a second volume, which is removed afterwards. The state file, with its
  history, mail summaries, and Gmail tokens, never leaves the Pi.
- A job still cannot reach a sink, a staging tool, or `job_*`. Moving
  `run_job` out of the channel keeps `report_only=True` on its surface.
- Gmail stays on `gmail.modify`. The draft lookup is a read. `_refuse_path`
  is unchanged.
- The `inbound` table holds sender numbers and timestamps, which the logs
  already hold. It holds no message text. Backups hold the same Gmail tokens
  as the live file, at mode 600 in the same volume.
- What the other person's assistant can learn: nothing new. Restart notices
  go only to the person whose turn was interrupted. Relay failure notices go
  only to the sender.

## Open questions

All closed by the owner on 2026-10-30.

1. **A message part that may or may not have arrived when the process was
   killed.** Answer: send it once more, after the line "Sent again after a
   restart. You may already have this part." This holds for every outbound
   message, relays included; for a relay the line goes in front of the
   stored bytes on the other phone. Reflected in: Shared pieces, "Uncertain
   parts"; stage 2's relay bullet; Risks, "Duplicates or gaps"; AC1.7 and
   AC2.5a; [decision 0004](../decisions/0004-relays-after-an-outage.md).
2. **How late may a scheduled relay go out after an outage?** Answer: up to
   one hour late. After that it is dropped and the sender is told.
   Reflected in: stage 2's late-limit bullet (`relay_late_limit_seconds`,
   3600); Risks, "Late messages after an outage"; AC2.5;
   [decision 0004](../decisions/0004-relays-after-an-outage.md).
3. **Edited messages.** Answer: an edit to a message already answered gets
   "I already answered the first version. Send the change as a new
   message." and nothing else. Reflected in: stage 3, Inbound; Risks,
   "Refusing edits"; AC3.7. An edit whose original was never answered runs
   as a new message.
4. **Confirmed calls that failed.** Answer: the summary offers a yes that
   retries only the failed calls. Reflected in: stage 3, Proposals (the
   retry proposal, built from the stored calls, opening when the summary
   has arrived); Security review; AC3.5.
5. **A yes interrupted by a restart.** Answer: if the gateway is back within
   ten minutes of the yes, the remaining calls finish automatically;
   otherwise it stops and says what ran. Reflected in: stage 3, Recovery
   adds (`commit_resume_seconds`, 600); Risks, "Resuming a yes"; Security
   review; AC3.3.
6. **Where the migration dry run runs.** Answer: on the Pi, against a copy
   in a second volume. The state file never leaves the Pi. Reflected in:
   Risks, "Migrating live rows" and "Rolling back the schema"; the dry-run
   steps under Rollout; Security review; AC0.6.

PRD conflicts: answers 1 and 2 change prd.md § Relay step 3. Decision 0004
records both, and prd.md § Relay is amended in the stage 2 commit that
ships them (AC2.10). No conflict remains open. Option C's "separate job
runner" stays an in-process task, so it does not conflict with the
architecture's scheduling decision.
