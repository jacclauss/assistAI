# 0004 — A relay after a restart or an outage

Date: 2026-10-30
Status: accepted

## Context

[prd.md § Relay](../prd.md) says that at the chosen time the gateway sends
"those same bytes", and that a send that already reached the phone is not
repeated. [Spec 018](../specs/018-durable-outbox.md) records each part of
every outbound Signal message before and after it is posted. Two cases are
still open after that.

First, Signal has no idempotency key. If the process is killed while one
part is being posted, the gateway cannot know whether that part reached the
phone. Sending it again may repeat it. Not sending it may drop it.

Second, the PRD does not say what happens when a scheduled relay cannot go
out at its time, for example because Signal or the Pi is down. Today the
runner retries forever and nobody is told (finding C-12). A message meant
for 8:00 could arrive in the afternoon.

## Decision

A part that may not have arrived when the process was killed is sent once
more, after the line "Sent again after a restart. You may already have this
part." This holds for every outbound message, relays included. On a relay
that line goes in front of the stored bytes on the other phone; the bytes
themselves are unchanged. A scheduled relay may go out up to one hour after
its time. After that it is dropped, and the sender is told it was not
delivered.

## Consequences

- The other phone can, rarely, see one part of a relay twice, labelled as
  sent again. It does not silently miss a part.
- A relay is never delivered more than an hour late. The sender is told
  after the third failed attempt that it has not gone out, and told again
  when it is dropped. That closes C-12 and matches "failures never fail
  closed with silence".
- Her assistant's record of the relay stores the body only, without the
  resend line.
- prd.md § Relay is amended in the same commit that ships the stage of spec
  018 that implements each part of this decision. Both the resend line on
  relays and the one-hour limit ship in stage 2, when relays move to the
  outbox. Until then the PRD is unchanged.
- Spec 018 tests: `tests/test_jobs.py::test_an_uncertain_relay_part_is_resent_after_the_marker`,
  `tests/test_jobs.py::test_a_relay_that_cannot_send_tells_the_sender`,
  `tests/test_jobs.py::test_a_relay_fifty_nine_minutes_late_still_goes_out`,
  and `tests/test_jobs.py::test_a_relay_past_its_late_limit_is_dropped_and_said`.
