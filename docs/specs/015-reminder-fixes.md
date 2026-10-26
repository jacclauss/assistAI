# 015 — Reminder and job fixes from the first independent review

Status: done

Implements: [prd.md § Jobs](../prd.md) and [prd.md § Relay](../prd.md).
Decision: [0001](../decisions/0001-reminders-stage-when-tainted.md).

## Outcome

The review of spec 014 found behaviour that breaks the PRD or `AGENTS.md`.
Afterwards:

- A reminder asked for right after the assistant read outside text waits
  for a yes. Whatever a job later writes into history keeps the label of
  the conversation that created it.
- A one-time job has to say whether it texts its sentence or writes a
  report. It is never silently a reminder.
- A watch that looked and found nothing stays quiet. A tool error is not an
  empty result.
- Saving a reminder does not erase the rest of the answer.
- A later message to the other phone is not sent, recorded, or confirmed
  twice when a bookkeeping write fails.
- A reminder cannot fire while the turn that saved it is still running.
- A reminder may mention scheduling a job. A digest that asks to make a job
  is still refused.

## Out of scope

Surviving a process restart between a send and its bookkeeping write. The
runner remembers sends in memory; a crash in that window can still repeat
one message. New tools. Gmail scope.

## Design

- `Job.tainted`, column `tainted`, `SCHEMA_VERSION` 14. The broker binds
  the conversation's taint alongside the acting agent while a handler runs.
  `job_create` and a future `relay` store it. A confirm runs with the
  proposal's taint as well as the history's.
- `BoundSurface.execute` saves a verbatim reminder without staging only
  when the scope is clean.
- The runner labels the reminder text, the relay confirmation, and a digest
  report untrusted when the job is tainted. A tainted digest still gets its
  prompt as an instruction: the person saw it and said yes.
- `parse_create` refuses `once` without an explicit `verbatim`.
- `empty_report_is_real` counts only report-tool results without an
  `error`. The "listed mail but said NONE" rule applies to schedules only.
- The channel sends the server's reminder sentence first, then the model's
  answer. The tool result tells the model the person already gets that
  sentence.
- The runner remembers relay sends, relay records, and report deliveries
  in memory until the matching row write succeeds, and retries the write
  instead of the side effect.
- The channel holds the ids of reminders saved by a turn in progress. The
  runner skips them until that turn ends.
- The another-job check skips verbatim reminders and also catches "make"
  and "add".

## Acceptance criteria

- [x] AC1 — A verbatim reminder in a tainted conversation stages; in a clean one it saves (`tests/test_broker.py::test_a_reminder_in_a_tainted_conversation_waits_for_yes`, `tests/test_broker.py::test_a_verbatim_reminder_is_saved_without_a_yes`)
- [x] AC2 — A job confirmed from a tainted proposal is stored tainted, and the preview and the summary after yes are stored untrusted (`tests/test_jobs.py::test_a_job_confirmed_in_a_tainted_conversation_remembers_it`)
- [x] AC3 — A tainted job's reminder text, relay confirmation, and report are stored untrusted, and so are its failure and expiry notices (`tests/test_jobs.py::test_a_tainted_reminder_is_stored_as_untrusted`, `tests/test_jobs.py::test_a_tainted_relay_confirmation_is_untrusted`, `tests/test_jobs.py::test_a_tainted_digest_report_is_untrusted`, `tests/test_jobs.py::test_a_tainted_job_failure_notice_is_untrusted`, `tests/test_jobs.py::test_a_tainted_watch_expiry_notice_is_untrusted`)
- [x] AC4 — A version 13 database keeps its job rows, they open clean, and a tainted job saves after the upgrade (`tests/test_store.py::test_schema_v13_gains_job_taint_and_keeps_rows`)
- [x] AC5 — A one-time job without `verbatim` is refused (`tests/test_jobs.py::test_a_one_time_job_must_say_whether_it_is_verbatim`)
- [x] AC6 — A watch that sees other mail and answers NONE stays quiet (`tests/test_jobs.py::test_a_watch_that_sees_other_mail_stays_quiet`)
- [x] AC7 — An inbox error followed by NONE is a failed check (`tests/test_jobs.py::test_a_failed_inbox_is_not_an_empty_report`)
- [x] AC8 — A reminder keeps the rest of the answer, with the server's sentence first; an answer that names another time is dropped (`tests/test_channel.py::test_a_reminder_keeps_the_rest_of_the_answer`, `tests/test_channel.py::test_the_sent_reminder_time_is_the_servers`)
- [x] AC9 — A failed mark does not repeat a relay send, a relay record, or a report (`tests/test_jobs.py::test_a_relay_whose_sent_mark_fails_is_not_sent_again`, `tests/test_jobs.py::test_a_relay_record_whose_mark_fails_is_not_written_again`, `tests/test_jobs.py::test_a_report_whose_delivered_mark_fails_is_not_sent_again`)
- [x] AC10 — A reminder saved in a turn or by a yes does not fire before that reply goes out (`tests/test_channel.py::test_a_reminder_does_not_fire_before_its_turn_ends`, `tests/test_jobs.py::test_a_reminder_confirmed_by_yes_waits_for_the_summary`)
- [x] AC11 — A reminder may mention scheduling a job; a digest that asks to make a job is refused (`tests/test_jobs.py::test_a_reminder_may_mention_scheduling_a_job`, `tests/test_jobs.py::test_a_digest_that_asks_to_make_a_job_is_refused`)
- [x] AC12 — Tests that would have missed a regression now catch it: migration of an existing row (`tests/test_store.py::test_schema_v12_keeps_an_existing_job_and_defaults_it_off`), the reminder gap (`tests/test_jobs.py::test_a_daily_reminder_can_start_fifty_days_out`), a relay `start_at` without the clock (`tests/test_broker.py::test_a_guessed_relay_start_at_is_not_staged`), the layering walker (`tests/test_invariants.py::test_the_import_walker_sees_every_import_form`), and zero eval trials (`tests/test_evals.py::test_zero_trials_is_refused`)
- [x] AC13 — The PRD, architecture, and release notes describe this behaviour, including that a job delivers a relay the person already confirmed (`tests/test_invariants.py::test_relative_doc_links_resolve`, manual read)

## Security review

- No tool is added or reclassified.
- Under taint, a reminder stages like every other write. The commit path
  runs with the proposal's taint, so the stored row says where it came from.
- A job still cannot reach a sink, a staging tool, or `job_*`. Delivering a
  relay the person confirmed is the runner's work, not the model's.
- The other person's assistant still sees a relay only as an untrusted
  record after the yes.

## Open questions

None. The taint question is closed by decision 0001.
