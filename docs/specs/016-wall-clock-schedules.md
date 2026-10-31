# 016 — Wall-clock schedules the server owns

Status: draft

Implements: [prd.md § Jobs](../prd.md), [prd.md § Relay](../prd.md),
[prd.md § Calendar](../prd.md), and the "Timezone for 'every morning at 7'"
item in [prd.md § Open engineering](../prd.md).
Audit: option B, findings C-1, C-2, C-3, C-7, C-11, X3-1, X3-3, X3-4, X3-5,
X3-7, B-2, B-7, B-8, F-6.

## Outcome

A daily job runs at the same local time every day, including across
daylight saving, after a slow run, and after a retry. "Every morning" with
no hour runs in the morning. Moving a job keeps its date and hour unless the
person asked to change them. The person names a time in the words they used
("tomorrow at 8", "Monday at 7:30", "December 7 at 9"). The server turns
that into an instant with its own clock. Every time the person sees, in a
tool result, a preview, or a confirmation, is written by one server
formatter in household local time. The model never converts a Unix number.
A time that does not exist because the clocks moved forward is refused with
the reason. A time that happens twice is the next one still ahead, and the
text says which. The gateway does not start Signal without a household time
zone.

Before this, the only active job on the Pi, the "mgmplus cancel reminder"
(daily, first run Thursday, December 10 at 8:00 AM CST), would have texted
at 9:00 AM from Sunday, March 14, 2027, and a few seconds later each day
before that.

## Out of scope

- How a job run's result is judged, stored in history, or fed to the next
  run, and the checks on outbound model text. That is spec 017. This spec
  supplies the formatter 017 uses for any time it checks.
- The reply clock regex (`_names_another_time`, X3-2). Spec 017 owns it.
  This spec keeps the server sentence in a shape that regex still reads.
- Proposal states, retrying a yes that failed partway, and durable send
  records. That is spec 018. A lapsed start time here is handled inside
  today's delete-then-run commit.
- Sending the job list to the phone as server text instead of model text.
  The data the model reads no longer holds a number to convert. Whether the
  server should also speak the list is a 017 question.
- Weekday rules ("weekdays at 7"), month rules, and multiple times per day.
  `every_days` 7 covers "every Thursday".
- The REPL. It binds no job tools and keeps its UTC `get_time`.
- D-1, D-7, D-11, G-1, G-2, G-3, G-5, H-4, H-5, A-5, F-5, F-7, X4-6, E-4,
  and C-14.

## Design

The architecture sections this changes are [Jobs](../architecture.md#jobs)
and [Staged actions](../architecture.md#staged-actions). No tool is added or
reclassified. Three tools change their time arguments.

### One household clock: `src/assistai/clock.py`

A new module holds every local-time rule. Nothing else calls `strftime`,
`datetime.fromtimestamp`, or `replace(tzinfo=...)` on a household time.

- `HouseholdClock`, a frozen dataclass: `zone: ZoneInfo` and
  `now: Callable[[], float]` (default `time.time`). `today()` is the local
  date. Built once at startup from `Settings.timezone` and passed to the
  broker, the channel, the job runner, `bind_jobs`, `bind_relay`,
  `bind_calendar`, and `format_proposal`. Tests build it with a fixed
  `now` in `America/Chicago`, which replaces the float lambdas the audit
  found in every job test.
- `LocalAt`, a frozen dataclass: `date: date | None`, `day: str | None`
  (`today`, `tomorrow`, or a weekday name), `hour`, `minute`, and
  `fold: int | None`.
- `Delay`, a frozen dataclass: `days`, `hours`, `minutes`.
- `Schedule`, a frozen dataclass: `zone`, `anchor_at` (the instant of the
  first scheduled run), and either `every_days` with `at_local` ("HH:MM"),
  or `every_seconds`. A one-time job has neither.
- `resolve_at(at, *, clock) -> float`. A `date` is used as given. A `day`
  is resolved against the local date: a weekday name is the next such day
  after today. With neither, it is the next time that clock time comes
  round, today if still ahead, otherwise tomorrow. A local time inside the
  spring-forward gap raises, naming the day and the jump ("2:30 AM does not
  exist on Sunday, March 14, 2027: clocks go from 2:00 to 3:00"). A local
  time that happens twice is the first occurrence unless `fold` says 1 or
  the first has already passed, in which case it is the second. A time
  more than 60 seconds in the past raises, and the message states the
  server's local now in words so the model can correct the day.
- `next_slot(schedule, *, after: float) -> float`. The earliest slot
  strictly after `after`. For `every_days`, slots are the local dates
  `anchor_date + k * every_days` at `at_local`, where `anchor_date` is the
  local date of `anchor_at`. A slot inside the spring gap is read with the
  offset before the change, so it runs one clock hour later that day only.
  A slot inside the fall-back hour runs at its first occurrence only. For
  `every_seconds`, slots are `anchor_at + k * every_seconds`. `k` is
  computed, then adjusted by at most two steps. No loop walks day by day.
- The formatter, used for every time a person or the model reads:
  - `say_when(instant, *, clock)`: "today at 8:00 PM", "tomorrow at 8:00 AM
    (Saturday, October 31)", "Tuesday at 7:30 AM (November 3)" within six
    days, "Thursday, December 10 at 8:00 AM" beyond that, with the year
    added when it is not the current one. When the local time is ambiguous
    because of fall-back, the zone abbreviation follows ("1:40 AM CST").
  - `say_repeat(schedule, *, clock)`: "every day at 8:00 AM", "every 2 days
    at 8:00 AM", "every Thursday at 8:00 AM", "every 90 minutes", "every 2
    hours".
  - `say_duration(seconds)`: "2 hours", "50 days". This moves from
    `relay._delay_phrase`.
  - `say_day(day, *, clock)`: "today, Friday, October 30", "yesterday,
    Thursday, October 29", "Monday, November 2".
- Clock errors are raised as `JobError` from jobs, wrapped as `RelayError`
  by relay and `CalendarError` by calendar, so `ToolRegistry.execute` keeps
  passing the message to the model unchanged.

### Startup and the system clock

- `Settings.timezone` becomes `str | None = None`. `Gateway._start_signal`
  refuses to start Signal when it is `None`, logs `gateway.timezone_missing`,
  and exits non-zero. The validator for a set value is unchanged. The REPL
  keeps working without it. `.env.example` uncomments
  `ASSISTAI_TIMEZONE=America/Chicago`. This closes B-8.
- New `Settings.job_default_time: str = "07:00"` (`ASSISTAI_JOB_DEFAULT_TIME`),
  validated as HH:MM. The value is open question 1.
- At start, the gateway logs `clock.zone` with the zone name, the local
  time, and the current UTC offset.
- Before the channel and the job runner start, `Gateway._wait_for_clock`
  waits for a believable clock, at most 120 seconds, checking the stop event
  every second so Docker's 10-second grace still applies. It asks the
  kernel whether the clock is synchronized (`adjtimex` with no changes; a
  `TIME_ERROR` state means not yet). It also compares `time.time()` with the
  newest past-event timestamp in the store (`messages.created_at`,
  `jobs.created_at`, `jobs.last_run_at`, `proposals.created_at`). A clock
  more than 5 minutes behind that is not believed. It logs
  `clock.waiting_for_sync` once, then `clock.synced`, or `clock.unsynced`
  when the 120 seconds run out and it starts anyway. If the kernel call is
  refused, it logs `clock.sync_unknown` and relies on the store check. A
  stored timestamp more than 24 hours ahead is logged as
  `clock.store_in_future` and ignored, so a bad row cannot hold the gateway
  down.
- `Store.take_proposal` also treats a proposal as expired when
  `now < created_at - 60`. A backward step can no longer keep a tainted
  proposal confirmable past its ten minutes. Together with the startup wait
  this closes the conversation wipe in X3-5, which needs the clock to start
  more than 14 days behind and then step forward.

### Stored schedule: `SCHEMA_VERSION` 15

`jobs` gains four nullable columns: `zone TEXT`, `anchor_at REAL`,
`every_days INTEGER`, and `at_local TEXT`. `Job` gains the same fields. The
schedule is `(zone, anchor_at, every_days, at_local)` for whole-day jobs
and `(zone, anchor_at, every_seconds)` for the rest. `next_run_at` stays,
indexed, as the cached next slot the runner polls. `every_seconds` stays and
is written as `every_days * 86400` for whole-day jobs, so an older image can
still step a row after a rollback.

- `Store.save_job` writes the new columns. `_job_from_row` reads them.
- `Store.advance_job(job_id, *, now)` reads the row inside the same
  transaction. If the stored `next_run_at` is already after `now`, a
  reschedule landed during the run and is kept. Otherwise it writes
  `next_slot(schedule, after=now)`. The SQL `? + every_seconds` is gone. The
  finish time, run length, and retry delay no longer move the next run. A
  job that missed slots while the Pi was down runs once, then returns to
  its slot. Skipped slots are logged as `jobs.slots_skipped` with a count.
- `Store.reschedule_job` writes the new schedule columns and `next_run_at`
  together. It still leaves the outbox alone.
- Migration. `_init_schema` adds the columns with the existing
  `_ensure_*` pattern, which is idempotent. The store does not know the
  zone, so `Store.backfill_job_schedules(zone) -> int` fills every row
  whose `zone` is NULL, active and cancelled alike. The gateway calls it on
  every start, before the runner. For each row:
  - `zone` is the household zone. `anchor_at` is `next_run_at`.
  - When `every_seconds` is a whole number of days, `every_days` is that
    number. `at_local` is the local time, to the minute, of the slot the
    person chose. For a row that has never run, that is `next_run_at`, which
    came from `start_at` or `in_seconds`. For a row that has run, it is
    `next_run_at - every_seconds`, the slot that last fired, so a daily
    job stepped across Nov 1 by 0.1.8 keeps its 8:00 and does not become
    7:00.
  - Otherwise the row stays elapsed with its `every_seconds`.
  - Each row is logged as `store.job_schedule_migrated` with the job id,
    `every_days`, `at_local`, and whether it is cancelled. Never the prompt.
  - The runner skips a row whose `zone` is still NULL and logs
    `jobs.unscheduled_row`. That can only happen between a rollback and the
    next start of this version.
- Rollback. A version 14 image refuses a version 15 file
  (`_init_schema`: "state database is newer than this build"). The new
  columns are nullable and every version 14 query names its columns, so a
  version 14 image works on this file once `user_version` says 14. This
  spec adds the `python -m assistai store` command group, which 017 and 018
  extend with their own downgrade steps and with `backup` and `check`.
  `python -m assistai store downgrade --to 14` sets it, and only
  when the file is exactly version 15 with the four columns and nothing
  else changed. Going forward again re-runs the idempotent migration and
  the backfill picks up any job the older image created.
- New command `python -m assistai store jobs [--next N] [--from DATE]`
  prints each active job's id, name, next N runs through `say_when`, and
  `say_repeat`. It prints no prompt text. It is the operator's check before
  and after each deploy.

### Runner

`JobRunner` takes `clock.now` as its clock. `_deliver` calls
`advance_job(job.id, now=finished)` as before; the grid does the rest.
`_defer` keeps its in-memory retry time and never touches `next_run_at`.
When `next_slot` shifts a slot out of the spring gap, it logs
`jobs.dst_shifted` with the job id and the slot.

### Time arguments of `job_create`, `relay`, and `job_reschedule`

`every_seconds`, `in_seconds`, and `start_at` leave the tool schemas. In
their place:

- `at`: an object with `hour` (0 to 23) and `minute` (0 to 59, default 0),
  and at most one of `day` (`today`, `tomorrow`, or a weekday) or `date`
  (YYYY-MM-DD). The description says to copy the person's words: "tomorrow"
  is `day`, "December 7" is `date`, a bare "at 8" is neither. `fold` is
  written by the server and described as such.
- `delay`: an object with `days`, `hours`, and `minutes`, at least one
  minute in total. Only for a duration the person said ("in two hours",
  "in 50 days").
- `every_days`: whole days between runs. The run keeps its local time
  across daylight saving. 7 is weekly.
- `every_minutes`: elapsed minutes between runs, at least 1. A multiple of
  1440 is refused with "use every_days so it keeps its local time".
- `once`, `verbatim`, `ttl_seconds`, `name`, `prompt`, `kind`, and `body`
  are unchanged.

`jobs.parse_create` returns `JobCreate` with `first: LocalAt | Delay | None`,
`every_days`, and `every_minutes`. The rules:

- A repeating job has exactly one of `every_days` or `every_minutes`. A
  `once` job has neither and needs `at` or `delay`. A verbatim reminder
  still needs `at` or `delay`.
- `every_days` with no `at` and no `delay` starts at the next
  `job_default_time`. The 9:12 PM "every morning" request starts tomorrow
  at 7:00 AM. This replaces the prompt sentence that produced C-2.
- `every_days` with `delay` repeats at the local time of that first run.
- `every_minutes` with no first time starts one interval from now, as today.
- `parse_relay_when` uses the same parser. A relay with no `at` or `delay`
  still sends at the yes.
- A call that carries `every_seconds`, `in_seconds`, or `start_at` is
  refused with the names that replace them, and logged as
  `jobs.legacy_arguments_refused`. Old tool calls stay in history for up to
  14 days, and the model may copy them. One refusal and one retry is the
  existing pattern ("If this call is rejected ... call job_create again").

`jobs.resolve_start_at` is removed. `bind_jobs` and `bind_relay` resolve
the first run with `resolve_at` or the delay, build a `Schedule`, and save
`zone`, `anchor_at`, `every_days`, `at_local`, and `every_seconds`.

`job_reschedule` takes `name` or `id`, plus at least one of `at`,
`every_days`, or `every_minutes`:

- Only a new repeat: the next run stays where it is, and the grid is
  re-anchored on it. Moving the Dec 7 reminder to weekly on Oct 30 keeps
  Dec 7.
- A new `at`: for a whole-day job, `at_local` becomes the new time. The
  next run is the date of the stored next run at that time, or the next day
  if that has passed, unless `at` names a `day` or `date`. For an elapsed
  job, `at` sets the next run and re-anchors the grid.
- The watch TTL check is unchanged. This closes C-3.

The `get_time` gate in `BoundSurface.execute` moves from `start_at` to
`at`. It still requires a successful `get_time` in the same turn, as the
PRD says. A `delay` is not gated. `get_time`'s result gains `weekday` and
`today` (the `say_day` text) beside `local` and `utc`.

#### Staging stores the resolved time

`format_proposal` runs at stage time; the yes runs later. "Tomorrow" in a
preview shown at 11:58 PM would mean a different day at a 12:03 AM yes. So
when the broker stages `job_create`, `relay`, or `job_reschedule`, it
stores a canonical copy of the call: `at.day`, an `at` with neither `day`
nor `date`, and the default morning time all become `at.date` plus `fold`
where it matters. Nothing else in the arguments changes. The staged tool
result returns the canonical arguments to the model, and the preview is
formatted from them. A `delay` stays a delay and the preview says it counts
from the yes. This is the "resolved call" the architecture already
describes, and what is shown is what runs.

#### A start time that passes before the yes

The handler knows it is running from a confirm through a new context
variable, `relay.active_commit()`, set by `BoundSurface.commit` beside the
agent and the taint. When the first run of a confirmed `at` has passed:

- A repeating job starts at its next slot. The `say` line says so ("8:00
  AM today had passed; first run tomorrow at 8:00 AM").
- A one-time reminder or relay that passed by no more than
  `staging_ttl_seconds` plus 60 seconds goes out at the yes, and the `say`
  line says the named time had passed. Open question 2.
- Anything older is refused with that sentence.

Outside a confirm, a past `at` is refused as before. This closes B-7
without changing how a proposal is consumed.

#### Named clock times sent as delays

`delay` is shaped like a spoken duration, which leaves the model less
reason to do arithmetic, and the `say` line shows the resolved clock time.
In shadow mode, `SignalChannel._converse` logs
`jobs.delay_with_named_clock` with the agent and tool when the inbound text
contains a clock time and the turn saved or staged a call with `delay` and
no `at`. Nothing is refused. After two weeks of logs the owner decides
whether that becomes a refusal.

### One formatter in every result, preview, and confirmation

| Where | Before | After |
| --- | --- | --- |
| `job_create` result | `next_run_at` float, `fires_at_local`, `local_now` | `first_run` (`say_when`), `repeats` (`say_repeat`), `say` for every job, not only verbatim ones |
| `job_list` result | `next_run_at`, `expires_at`, `every_seconds` | `next_run`, `repeats`, `ends` for a watch. No Unix number reaches the model |
| `job_reschedule` result | `next_run_at` float | `next_run`, `repeats`, and a `say` sentence that `format_executed` shows after yes |
| `job_cancel` result | id and name | adds a `say` sentence naming the run that will not happen |
| `relay` future result | `say` from `describe_saved_reminder` | the same, through `say_when` |
| `describe_saved_reminder` | its own `strftime` | `say_when`. The sentence keeps "8:00 AM" spelled so the X3-2 regex still matches it |
| `_format_job_create` | "Every: 86400 seconds", "First run" only for `start_at` | "First run: tomorrow at 7:00 AM (Saturday, October 31)" or "First run: 2 hours after you reply yes", and "Repeats: every day at 7:00 AM" or "Once" |
| `_job_create_line` (bundles) | interval only | the same first run and repeat text as the single preview |
| `_format_job_reschedule`, `_job_reschedule_line` | interval only | "Next run: ..., then every ..." |
| `relay.when_phrase` | raw `start_at`, delay phrase | `say_when` for `at`, `say_duration` plus "after you reply yes" for `delay` |
| `CalendarDraft.label` | naive `strftime` | `say_when` for start, end time, and the abbreviation for an ambiguous hour |
| `calendar_today` result | `date` | adds `day` from `say_day`, so a wrong date reads as "yesterday, Thursday, October 29" to the model and in its answer (F-6) |
| `agents._local_clock` | its own `strftime` | the clock's local text and weekday |

`format_proposal(calls, *, tainted, clock)` and
`CalendarDraft.label(clock)` take the clock. A new invariant test fails if
`strftime`, `fromtimestamp`, or `replace(tzinfo=` on a household time
appears outside `clock.py`.

### Calendar adds

- `parse_add(arguments, *, clock)` attaches the zone itself. A start or end
  inside the spring gap is refused with `resolve_at`'s reason. A start in
  the fall-back hour is the first occurrence, and the label names it
  ("1:30 AM CDT"). The default end is one real hour after the start
  instant, not one naive hour. An explicit end is compared as an instant.
- The ten-year bound uses `clock.today()`, not `date.today()`. This closes
  X3-7. Callers in `calendar/tools.py`, `staging.py`, and `broker.py` pass
  the clock.
- `render_event` writes the aware instants it is given. This closes the
  calendar half of X3-1 (F-3).

### Prompts and tool descriptions

- `agents.system_prompt_for` loses "A repeating check with no named hour
  ... is verbatim false, every_seconds 86400, and no start_at" and the
  `start_at` and `in_seconds` sentences. It gains: copy the person's words
  into `at` (`day` for "tomorrow" or a weekday, `date` for a calendar date,
  neither for a bare hour); use `delay` only for a duration they said; a
  daily job is `every_days` 1 and keeps its time; a daily check with no hour
  omits `at`.
- `JOB_CREATE_SPEC`, `JOB_RESCHEDULE_SPEC`, `RELAY_SPEC`, and
  `GET_TIME_SPEC` change to match. The relay example "in_seconds is 120"
  becomes `delay` with `minutes` 2.

### Eval harness

`evals/harness.py` gains two case fields and one expectation:

- `now` and `timezone`. The case's `get_time` returns that clock's real
  shape, and `system_prompt_for` gets the timezone, so the prompt clock and
  the tool clock agree. Today's fixture is `{"ok": true}`, which passes the
  gate with no clock.
- `[expect.arguments.<tool>]`, dotted paths compared to the first call of
  that tool, staged or run. Today a case grades tool names only.

New or changed cases, all at `America/Chicago`:

- `reminder-named-clock`: "remind me at 8:53pm to call mom" at 3:04 PM.
  Expects `job_create` with `at.hour` 20, `at.minute` 53, and no `delay`.
- `reminder-tomorrow-late-night`: "remind me tomorrow at 8 to take the
  bins in" at 2026-10-30T23:30. Expects `at.day` "tomorrow" or `at.date`
  "2026-10-31".
- `job-morning-email` (changed): expects `every_days` 1 and no `delay`.
- `job-dated-digest`: "Starting December 9, check flight prices to Denver
  every morning". Expects `at.date` "2026-12-09".
- `job-list-when`: a `job_list` fixture with the mgmplus row in the new
  shape. "When is my mgmplus reminder?" must contain "December 10" and
  "8:00 AM", and must not contain "2:00 PM" or "14:00".
- `job-reschedule-hour`: "Move my mgmplus reminder to 9am". Expects
  `job_reschedule` with `at.hour` 9 and no `every_days`.

### Docs changed with the code

- `docs/architecture.md` § Jobs: the stored schedule, `at` and `delay`,
  whole days on the local clock, DST rules, lapsed start times, and the
  required timezone. § Staged actions: the stored call has its relative
  time resolved.
- `docs/pi-deploy.md`: `ASSISTAI_TIMEZONE` is required; the
  `store jobs` check; a "Roll back a schema change" section.
- `docs/release-notes.md` for each deploy.
- `.env.example`: the timezone line is uncommented.
- The PRD is unchanged unless open question 4 or 5 says otherwise.

## Risks and mitigations

The audit rated this option low to moderate: a `SCHEMA_VERSION` bump, a
migration of live rows, DST tests, and changes contained to jobs, relay,
and the staging formatters. These are those risks and the ones found while
writing this spec.

**The migration puts a live job at the wrong hour.** The one active row is
the mgmplus reminder. A wrong `at_local` would text at the wrong time every
day until noticed.

- The previous-slot rule keeps a stepped daily row on its chosen hour.
  `test_store.py::test_a_legacy_daily_row_that_ran_keeps_its_hour` builds a
  row that 0.1.8 stepped across Nov 1 and asserts 8:00, not 7:00.
- `test_store.py::test_schema_v14_gains_wall_clock_schedules_and_keeps_rows`
  opens a version 14 file shaped like the Pi's: the mgmplus row (86,400
  seconds, never run, next run 2026-12-10 08:00 CST), the cancelled morning
  email row (86,400 seconds, last run Oct 30, cancelled Oct 30), a cancelled
  row with an outbox, a 60-second watch, and a delayed relay. It asserts
  every row survives, cancelled rows stay cancelled and gain a zone, the
  mgmplus row gets `every_days` 1 and `at_local` 08:00 with an unchanged
  `next_run_at`, and the watch stays elapsed.
- Before deploy B, the operator copies the Pi's state volume (the tar
  command in `pi-deploy.md`), opens the copy with the new image, and runs
  `store jobs --next 3` and `store jobs --from 2027-03-12 --next 4`. Both
  must show 8:00 AM, including March 14 and 15, 2027.
- After deploy, `store.job_schedule_migrated` must appear once per row,
  and `store jobs` on the Pi must match the rehearsal.

**An older image cannot open the newer database.** Rollback is the previous
image tag, and a version 14 image refuses version 15.

- The columns are additive and nullable, and `every_seconds` is still
  written, so `store downgrade --to 14` is enough for 0.1.8 to run.
  `test_store.py::test_downgrade_lets_a_v14_writer_add_a_job_and_forward_again_backfills_it`
  runs the command, inserts a row with 0.1.8's exact column list, reopens
  with this version, and asserts the row is backfilled.
- `test_store.py::test_downgrade_refuses_a_file_it_does_not_recognise`.
- During the deploy B rehearsal the operator also runs the rollback on the
  copy and opens it with the 0.1.8 image
  (`python -c "from assistai.store import Store ..."` listing jobs). Deploy
  B does not go ahead until that works.
- While rolled back, 0.1.8 steps jobs the old way. On return, the first
  advance puts them back on their slot.

**The model handles the new arguments badly.** Reminders could fail more
often for a few days, and old `start_at` calls in history invite copying.

- Refusals name the replacement fields, which takes one retry.
- The six eval cases above must pass at their `pass_rate` over ten trials,
  and `make compare` must score the new schemas, before deploy C.
- Before deploy C, every `job_create` and `relay` call in the Pi's stored
  history (the `tool_calls` column of the copied file) is replayed through a
  throwaway script that maps its old arguments to the new ones. Each must be
  expressible. This checks the new arguments cover what the household has
  actually asked for.
- After deploy, watch `jobs.legacy_arguments_refused` and `tool.failed`
  with `name=job_create` or `name=relay`. They should fall to zero within
  the 14-day history window. If reminders fail outright, roll back to the
  deploy B image, which shares the schema.

**A proposal staged by the old version is confirmed by the new one.** Its
stored call has `start_at` or `in_seconds`.

- The commit refuses the old shape with "this was proposed before an
  update; ask again", which shows under the commit summary.
- The operator deploys when the `proposals` table is empty. Proposals
  expire after ten minutes, so this means waiting at most that long.

**The default morning hour surprises someone.** C-2's fix changes what
"every morning" means.

- The preview names the hour before the yes. The release note and the
  household message below say what the default is.
- The value is open question 1, and it is a setting.

**Rewriting the staged call weakens "what you approved is what runs".**

- Only time fields change, by a pure function of those fields and the
  server clock. The canonical call is what is stored, shown, and run.
  `test_staging.py::test_a_relative_day_is_fixed_when_staged` stages
  "tomorrow" at 11:58 PM, confirms at 12:03 AM, and asserts the run date is
  the one in the preview.
- `test_broker.py::test_canonicalising_changes_only_time_fields` compares
  every other argument byte for byte.

**A DST rule is wrong, or the image's zone data is.**

- `test_clock.py` covers both 2026-11-01 and 2027-03-14 in
  `America/Chicago`: a daily 8:00 job, a daily 1:30 job, a daily 2:30 job,
  an every-90-minutes job, an `at` in the gap, and an `at` in the repeated
  hour.
- `test_clock.py::test_next_slot_holds_its_rules_for_many_schedules` runs a
  seeded loop over 2,000 schedules from 2026 to 2030. Each slot is after
  `after`, keeps `at_local` except on a gap day, and appears once on a
  fall-back day. No new dependency.
- `test_clock.py::test_the_zone_database_knows_the_2026_and_2027_transitions`
  fails loudly if zone data is missing.
- On the Pi after deploy A:
  `docker compose ... exec gateway python -c` printing the UTC offsets of
  2026-11-01 01:30 with fold 0 and fold 1 (−05:00 and −06:00) and of
  2027-03-14 03:30 (−05:00).

**The startup clock wait holds the gateway down.**

- The wait is capped at 120 seconds and honours shutdown. A stored
  timestamp more than 24 hours ahead is ignored.
  `test_gateway.py::test_signal_waits_for_a_synced_clock_then_starts` and
  `test_gateway.py::test_a_future_row_does_not_hold_startup`.
- Watch `clock.unsynced` and `clock.sync_unknown` after each restart.

**Nothing in this spec suppresses a message without saying so.** Every new
refusal (gap time, legacy argument, whole-day minutes, unset timezone)
returns a reason the model passes on, or stops startup with a log line. The
one check that could suppress a real reminder, a delay beside a clock time,
ships as log only. That is why no replayed-history eval is needed to tune
this spec. The history replay above checks expressiveness instead.

**Tests that must exist and fail before the change.** They land first as
strict `xfail` in commit 1, so `make check` passes and each one proves the
bug. The fixing commit removes the mark. They are the zoned-clock tests in
report section 6, item 2: daily 8:00 across fall-back (C-1), "every
morning" at 21:12 (C-2), reschedule keeps the next date (C-3), `job_list`
reports local time (X3-3), a spring-gap calendar add (X3-1), 1:40 AM after
fall-back (X3-1), the year bound on Dec 31 at 7:00 PM CST (X3-7), and the
UTC default (B-8).

**Pinned tests that change, on purpose:**

- `test_jobs.py::test_a_slow_run_does_not_make_the_job_immediately_due`
  expects 121 instead of 140. The property in its docstring still holds:
  the next run is not in the past. It is now on the job's grid, not the
  finish time plus 60. The same expectation at `test_jobs.py:612`, `:841`,
  `:961`, `:1013`, and `:1040`, the 70.0 in
  `test_quiet_watch_runs_when_the_send_budget_is_gone` and
  `test_pending_notify_survives_reopen`, and the 105.0 in
  `test_unpersisted_notify_retries_history_only` move to grid slots for the
  same reason.
- `test_config.py::test_defaults_are_conservative` asserts the timezone
  default is unset.
- `test_agents.py::test_example_roster_loads` drops the `start_at`,
  `in_seconds`, "in_seconds 120", and "every morning" substrings. The eval
  cases above replace them, as TT4-12 asked.
- `test_staging.py::test_a_job_create_preview_shows_the_stored_schedule`
  and `::test_cancel_then_create_preview_names_the_jobs` assert the first
  run and repeat text.
- `test_broker.py::test_a_guessed_start_at_is_not_saved`,
  `::test_in_seconds_is_counted_by_the_server`, and
  `::test_a_guessed_relay_start_at_is_not_staged` use `at` and `delay`
  with the same properties.
- `test_channel.py::test_the_sent_reminder_time_is_the_servers` matches the
  new wording.
- The remaining job, relay, and store tests that pass `every_seconds`,
  `in_seconds`, or `start_at` change arguments only.

**The household relies on today's behaviour.** After deploy C, Jacob sends
his spouse something like: "The assistant now keeps daily reminders at the
same time across daylight saving. 'Every morning' with no time means 7:00
AM. Confirmations now say the day, like 'tomorrow at 8:00 AM (Saturday,
October 31)'. A time that doesn't exist when the clocks spring forward gets
a question back." The release notes say the same.

## Rollout

Each commit is one logical change, updates the doc it changes, and passes
`make check`. Three deploys, each with its own rollback and closing check.

| # | Commit | Deploy |
| --- | --- | --- |
| 1 | Add `America/Chicago` clock fixtures and the strict-xfail tests across 2026-11-01 and 2027-03-14 so each time bug is proven before it is fixed. | A |
| 2 | Add `clock.py`: the household clock, `resolve_at`, `next_slot`, and the formatter, with its own tests and no callers. | A |
| 3 | Refuse to start Signal without a timezone, log the zone, wait for a believable clock, and expire a proposal after a backward step. | A |
| 4 | Store wall-clock schedules in schema 15, backfill live rows from their chosen slot, and add `store jobs` and `store downgrade`. Until commit 6, `job_create` maps a whole-day `every_seconds` to `every_days`. | B |
| 5 | Advance every job to its next slot so runs keep their local time across DST, slow runs, and retries. | B |
| 6 | Take named times as `at` and durations as `delay`, refuse the old arguments, start a daily job with no hour at the default time, and store staged times resolved. | C |
| 7 | Keep a job's next run and hour when it is rescheduled. | C |
| 8 | Format every time in tool results, previews, and confirmations with the one formatter, and drop Unix times from what the model reads. | C |
| 9 | Start a confirmed job whose time passed during the yes at its next slot, or at the yes for a one-time send. | C |
| 10 | Refuse calendar times in the spring gap, name the repeated hour, and use the household date for the year bound. | C |
| 11 | Grade eval arguments against a real clock, add the time cases, and log delays beside named clocks. | C |

**Deploy A (commits 1 to 3).** No schema change. Rollback is the previous
image tag. Closed when the `clock.zone` line on the Pi shows
`America/Chicago` and the expected offset, `clock.synced` appears after a
`docker compose restart gateway`, the DST offset check above prints the
right offsets, and a text to the bot is answered.

**Deploy B (commits 4 and 5).** Before: the state copy rehearsal (migration
and rollback), `make check`, and an empty `proposals` table. After:
`store.job_schedule_migrated` once per row, and `store jobs --next 3`
showing the mgmplus reminder on Thursday, December 10 at 8:00 AM, then
Friday and Saturday at 8:00 AM. Rollback:
`docker compose ... stop gateway`, then
`docker compose ... run --rm --no-deps gateway store downgrade --to 14`,
then load the deploy A image and `up -d --no-build`. Closed when `state
jobs` matches the rehearsal. There is no Nov 1 deadline for this deploy:
the morning email job is cancelled and the mgmplus reminder first runs
December 10. The deadline is March 14, 2027, or the next daily job someone
creates, whichever comes first.

**Deploy C (commits 6 to 11).** Before: `make eval` with the six time cases
at their pass rates, `make compare`, the history replay, and an empty
`proposals` table. After, from Jacob's phone: "remind me in 2 minutes to
test 016" arrives on time and the confirmation names today and the minute;
"remind me tomorrow at 7:05 to test the morning" names tomorrow, the
weekday, and 7:05 AM; "when is my mgmplus reminder?" names December 10 at
8:00 AM; "move the test reminder to 7:10" previews the next run; then
cancel both tests. Watch `jobs.legacy_arguments_refused`,
`jobs.delay_with_named_clock`, `jobs.slots_skipped`, `jobs.dst_shifted`,
and `tool.failed` for a week. Rollback is the deploy B image, which reads
the same schema. Closed after the phone checks pass, the next morning's
7:05 test fires at 7:05, and the household message is sent.

## Acceptance criteria

- [ ] AC1 — A daily 8:00 job stays at 8:00 local across 2026-11-01 and
  2027-03-14, and a slow run or a retry does not move the next run.
  Closes C-1. (`tests/test_clock.py::test_daily_eight_am_stays_eight_am_across_fall_back`,
  `tests/test_clock.py::test_daily_eight_am_stays_eight_am_across_spring_forward`,
  `tests/test_jobs.py::test_a_late_run_and_a_retry_do_not_move_the_next_run`)
- [ ] AC2 — "Every morning" with no hour, asked at 21:12, first runs at the
  default time the next morning; the model omits the hour and uses
  `every_days` 1. Closes C-2.
  (`tests/test_jobs.py::test_every_morning_without_an_hour_runs_at_the_default_time`,
  `evals/cases/job-morning-email.toml`)
- [ ] AC3 — A reschedule that changes only the repeat keeps the next run
  and its hour; one that changes the hour keeps the date. Closes C-3.
  (`tests/test_jobs.py::test_reschedule_keeps_the_next_run_and_its_hour`,
  `tests/test_jobs.py::test_reschedule_with_at_moves_the_time_of_day`,
  `evals/cases/job-reschedule-hour.toml`)
- [ ] AC4 — A digest whose start date was only in its prompt shows its real
  first date in the preview before the yes, and the model puts a named
  start date in `at.date`. Closes C-7.
  (`tests/test_staging.py::test_a_digest_preview_names_its_first_date`,
  `evals/cases/job-dated-digest.toml`)
- [ ] AC5 — A named time without `get_time` in the turn is refused; a
  named clock goes to `at`, not `delay`; a delay beside a clock time in the
  person's message is logged. Narrows C-11 to model arithmetic that the
  `say` line shows; enforcement waits on the logs.
  (`tests/test_broker.py::test_a_named_time_without_get_time_is_refused`,
  `tests/test_channel.py::test_a_delay_beside_a_named_clock_is_logged`,
  `evals/cases/reminder-named-clock.toml`)
- [ ] AC6 — A time in the spring gap is refused with the reason for jobs,
  relays, and calendar adds; a repeating slot in the gap runs an hour later
  that day only; a slot in the fall-back hour runs once; "1:40" at 1:20 AM
  CST on Nov 1 is twenty minutes away; a calendar event's default end is
  one real hour. Closes X3-1 and its calendar twin F-3.
  (`tests/test_clock.py::test_at_in_the_spring_gap_is_refused_with_the_reason`,
  `tests/test_clock.py::test_a_slot_in_the_spring_gap_runs_an_hour_later_that_day_only`,
  `tests/test_clock.py::test_a_slot_in_the_fall_back_hour_runs_once`,
  `tests/test_jobs.py::test_a_reminder_at_one_forty_after_fall_back_is_twenty_minutes_away`,
  `tests/test_calendar.py::test_add_in_the_spring_gap_is_refused`,
  `tests/test_calendar.py::test_add_in_the_fall_back_hour_says_which_one`,
  `tests/test_calendar.py::test_default_end_is_one_real_hour`)
- [ ] AC7 — `job_list` and `job_reschedule` return local times in words and
  no Unix number, and the model's answer about a job's time uses them.
  Closes X3-3. (`tests/test_jobs.py::test_job_list_reports_local_time_not_unix`,
  `tests/test_jobs.py::test_reschedule_result_names_the_next_run`,
  `evals/cases/job-list-when.toml`)
- [ ] AC8 — "Tomorrow at 8" at 11:30 PM local is the local tomorrow; a
  relative day is fixed at stage time so a yes after midnight runs the
  previewed date; an explicit date is shown with its weekday and relative
  day before or as it saves. Closes the 11:30 PM case of X3-4; a wrong
  explicit date is visible, not silent.
  (`tests/test_clock.py::test_tomorrow_at_eleven_thirty_pm_is_the_local_tomorrow`,
  `tests/test_staging.py::test_a_relative_day_is_fixed_when_staged`,
  `evals/cases/reminder-tomorrow-late-night.toml`)
- [ ] AC9 — A backward clock step expires a proposal, and Signal waits,
  bounded, for a believable clock. Closes the remaining X3-5 impact.
  (`tests/test_store.py::test_a_backward_clock_step_expires_a_proposal`,
  `tests/test_gateway.py::test_signal_waits_for_a_synced_clock_then_starts`,
  `tests/test_gateway.py::test_a_future_row_does_not_hold_startup`)
- [ ] AC10 — The calendar year bound uses the household date on Dec 31 at
  7:00 PM CST. Closes X3-7.
  (`tests/test_calendar.py::test_year_bound_uses_the_household_date`)
- [ ] AC11 — Every job and relay preview, single or bundled, names the
  first run, a delay says it counts from the yes, and a reschedule preview
  names the next run. Closes B-2 and its twins D-9, X3-6, and B-6.
  (`tests/test_staging.py::test_every_job_preview_names_the_first_run`,
  `tests/test_staging.py::test_a_reschedule_preview_names_the_next_run`)
- [ ] AC12 — A confirmed repeating job whose first time passed during the
  yes starts at its next slot; a one-time reminder or relay that passed
  within the proposal window goes out at the yes and says so; an older one
  is refused with the reason. Closes B-7.
  (`tests/test_channel.py::test_a_lapsed_daily_job_starts_at_its_next_slot`,
  `tests/test_channel.py::test_a_lapsed_one_time_reminder_is_sent_at_the_yes`,
  `tests/test_relay.py::test_a_lapsed_relay_is_sent_at_the_yes`)
- [ ] AC13 — The timezone has no default, and Signal does not start
  without one. Closes B-8.
  (`tests/test_config.py::test_defaults_are_conservative`,
  `tests/test_gateway.py::test_signal_does_not_start_without_a_timezone`)
- [ ] AC14 — `calendar_today` names the day it read relative to today.
  Closes F-6 as far as code can: a wrong date is a request for another day
  that the result now labels.
  (`tests/test_calendar.py::test_calendar_today_names_the_day_relative_to_today`)
- [ ] AC15 — A version 14 file shaped like the Pi's upgrades with every row
  kept, the mgmplus row at `every_days` 1 and 08:00, cancelled rows intact,
  and a stepped daily row on its original hour; rollback to 14 and forward
  again backfills a row the older version wrote.
  (`tests/test_store.py::test_schema_v14_gains_wall_clock_schedules_and_keeps_rows`,
  `tests/test_store.py::test_a_legacy_daily_row_that_ran_keeps_its_hour`,
  `tests/test_store.py::test_downgrade_lets_a_v14_writer_add_a_job_and_forward_again_backfills_it`,
  `tests/test_store.py::test_downgrade_refuses_a_file_it_does_not_recognise`)
- [ ] AC16 — Household local times are formatted only in `clock.py`.
  (`tests/test_invariants.py::test_local_times_are_formatted_in_one_place`)
- [ ] AC17 — `every_seconds`, `in_seconds`, and `start_at` are refused with
  the names that replace them, a staged call from before the update is
  refused at commit with "ask again", and `every_minutes` that make whole
  days are refused. (`tests/test_jobs.py::test_legacy_time_arguments_are_refused_with_the_new_names`,
  `tests/test_channel.py::test_a_proposal_from_before_the_update_asks_again`,
  `tests/test_jobs.py::test_minutes_that_make_whole_days_are_refused`)
- [ ] AC18 — A job missed while the Pi was down runs once, then returns to
  its slot. (`tests/test_jobs.py::test_a_job_missed_while_down_runs_once_then_returns_to_its_slot`)
- [ ] AC19 — Staging changes only time fields when it resolves a call.
  (`tests/test_broker.py::test_canonicalising_changes_only_time_fields`)
- [ ] AC20 — `next_slot` holds its rules over 2,000 seeded schedules, and
  the zone data knows the 2026 and 2027 transitions.
  (`tests/test_clock.py::test_next_slot_holds_its_rules_for_many_schedules`,
  `tests/test_clock.py::test_the_zone_database_knows_the_2026_and_2027_transitions`)
- [ ] AC21 — The six time eval cases pass at their `pass_rate` over ten
  trials. (`make eval` on the cases named above)
- [ ] AC22 — Architecture, Pi deploy, release notes, and `.env.example`
  describe this behaviour. (`tests/test_invariants.py::test_relative_doc_links_resolve`,
  manual read)
- [ ] AC23 — Each deploy's closing check in Rollout passes on the Pi.
  (manual, recorded in the PR)

## Security review

- No tool is added or reclassified. `job_create`, `job_reschedule`,
  `job_cancel`, and `relay` stay `staging`; `relay` keeps sink
  `message:other_peer`. `job_list` and `get_time` stay trusted and not
  staging. `calendar_today` stays untrusted; `calendar_add` stays staging
  with sink `calendar:write`. None has `web`.
- Staging now stores a resolved copy of the call. The broker rewrites only
  `at` from a relative form to a date and fold, using the server clock.
  The person is shown that copy, and the yes runs it. The model's original
  call stays in history. AC19 holds the rewrite to time fields.
- Under taint, nothing changes. A verbatim reminder still stages in a
  tainted conversation (decision 0001), and the stored row still records
  `tainted`. The new `say` lines are server text built from the stored row,
  but they quote job names and prompts the model wrote, so they keep the
  turn's untrusted label exactly as today's previews and summaries do.
- A report-only job still cannot reach `job_*`, `relay`, or any sink.
  `_not_report_tool` is unchanged.
- The other person's assistant learns nothing new. `job_list` is still
  scoped to the acting agent. A relay record still holds only the sender
  and body.
- No secret is involved. The new log events carry job ids, names, local
  times, and counts, never prompts or bodies. `store jobs` prints names and
  times only. `store downgrade` refuses a file it does not recognise.
- The `adjtimex` call reads clock state and changes nothing. If the
  container's seccomp profile refuses it, startup logs and goes on.

## Open questions

Only the owner can close these. Status stays `draft` until they are.

1. **Default time for a daily job with no named hour.** This spec proposes
   7:00 AM as `ASSISTAI_JOB_DEFAULT_TIME`. The 0.1.8 morning email ran at
   8:00. Which hour, and should the same default apply to the spouse?
2. **A one-time reminder or relay whose time passes while waiting for the
   yes.** Proposed: send it at the yes if it is less than eleven minutes
   late, and say the named time had passed. The alternative is to refuse
   and ask again. Which?
3. **Times the clocks skip or repeat.** Proposed: refuse a time in the
   spring-forward gap when a reminder, relay, or calendar event is created,
   and ask for another; take the next occurrence still ahead of a repeated
   time and name the zone. A repeating job's slot in the gap runs an hour
   later that day. The alternative for creation is to move a gap time to
   3:00 AM and say so. Which?
4. **Keep the `get_time` gate for `at`?** The PRD says "A named clock time
   is refused until the assistant has read the clock in that turn". With
   the server resolving "tomorrow" and weekday names itself, the gate adds a
   tool round and no check. This spec keeps it, as the PRD requires.
   Dropping it for relative days is a PRD change and needs a decision file.
5. **PRD wording on repeat gaps (possible conflict).** PRD § Jobs says "the
   gap between repeats can be any future value the clock can represent".
   Under this spec, a whole-day gap follows the local clock, so it is 23 or
   25 hours on a DST day, and a gap in minutes that makes whole days is
   refused in favour of `every_days`. I read that as within the PRD's
   intent, since the PRD lists "every morning at 7" as open engineering. If
   the owner reads "gap" as a fixed number of seconds, this is a conflict,
   and closing it needs a decision file and a PRD edit.
6. **Who tells the household.** Proposed: release notes, and Jacob texts
   his spouse after deploy C. Should the bot send that note itself? That
   would need a one-time outbound notice, whose durable record belongs to
   spec 018.
