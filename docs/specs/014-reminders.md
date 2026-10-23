# 014 — Reminders and later messages

Status: done

Implements: [prd.md § Jobs](../prd.md) and [prd.md § Relay](../prd.md)

## Outcome

Saying "remind me" stores the sentence and texts it at the time the server
saved. A repeating reminder is silent until that first time, then repeats
after the gap. A message to the other person still waits for one yes. A time
sends those stored bytes then, with no second yes. A mail check that never
opens the inbox is a failed check, not an empty inbox.

## Out of scope

A diagnostic log of the start time the model sent. Attachment bytes on a
relay. Changing Gmail scope.

## Design

No new tool. `job_create` and `relay` stay staged. A verbatim reminder is the
exception: `BoundSurface.execute` saves it immediately, including while the
conversation is tainted, because the yes was what dropped it. A digest still
waits for a yes.

The first run is `in_seconds` (added to the server clock) or `start_at` (local
ISO-8601 with no timezone suffix, and only after `get_time` in that turn).
`every_seconds` is the gap after that first run. A reminder without either
first time is rejected. A prompt that tells the job to create another job is
rejected. Jobs stay report-only.

`jobs` gains `once`, `verbatim`, `relay`, `relay_sent`, and `relay_recorded`.
`SCHEMA_VERSION` is 13. Older files gain the columns with default 0.

A schedule that answers NONE without a report tool, or after the inbox tool
listed mail, is asked to look once more and then reported as a failed check.
An empty result after a real look still messages.

## Acceptance criteria

- [x] AC1 — A verbatim reminder is stored without a yes (`tests/test_broker.py::test_a_verbatim_reminder_is_saved_without_a_yes`)
- [x] AC2 — A named time without `get_time` in the turn is refused (`tests/test_broker.py::test_a_guessed_start_at_is_not_saved`)
- [x] AC3 — The phone is told the time the server stored (`tests/test_channel.py::test_the_sent_reminder_time_is_the_servers`)
- [x] AC4 — A failed turn cancels the reminder it saved (`tests/test_channel.py::test_a_failed_turn_cancels_the_reminder_it_saved`)
- [x] AC5 — The first run can be months ahead, and the gap after it is separate (`tests/test_jobs.py::test_a_daily_reminder_can_start_fifty_days_out`)
- [x] AC6 — A reminder with no first time is rejected (`tests/test_jobs.py::test_a_reminder_without_a_first_time_is_rejected`)
- [x] AC7 — A job cannot be an instruction to create another job (`tests/test_jobs.py::test_a_job_cannot_be_an_instruction_to_create_another_job`)
- [x] AC8 — A delayed relay is not saved or sent until yes (`tests/test_broker.py::test_a_delayed_relay_is_not_saved_until_yes`)
- [x] AC9 — A send that already reached the phone is not repeated; the history record is retried on its own (`tests/test_jobs.py::test_a_relay_that_dies_after_send_is_not_sent_again`, `tests/test_jobs.py::test_a_relay_record_is_retried_without_sending_again`)
- [x] AC10 — NONE without a look is a failed check, not an empty inbox (`tests/test_jobs.py::test_an_unchecked_schedule_is_a_failed_check`, `tests/test_jobs.py::test_a_digest_that_never_looks_is_not_an_empty_report`)
- [x] AC11 — A previous schema opens and the new columns default off (`tests/test_store.py::test_schema_v4_gains_pending_notify_columns`, `tests/test_store.py::test_schema_v12_gains_relay_recorded`)

## Security review

- No tool was added or reclassified. `relay` remains `staging` and sink
  `message:other_peer`. `job_create` remains `staging`. The verbatim path
  skips the yes. It does not skip the broker.
- Under taint, a relay still stages. A verbatim reminder still saves. The
  stored sentence is the user's own text. The Signal reply shows it
  immediately, with the name to cancel. Marking that sentence untrusted
  would taint the next window, which job reports are forbidden to do.
- A report-only run cannot reach `relay`, `job_create`, or any other staging
  tool. `tests/test_jobs.py::test_job_run_cannot_stage_a_relay` and
  `tests/test_broker.py::test_report_only_surface_hides_relay_and_jobs`.
- The other person's assistant learns a relay only after the yes, as an
  untrusted record labelled from the sender. She does not see his reminders.

## Open questions

None.
