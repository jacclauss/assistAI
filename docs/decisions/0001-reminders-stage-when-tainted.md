# 0001 — A reminder waits for a yes when the conversation is tainted

Date: 2026-10-24
Status: accepted

## Context

[Spec 014](../specs/014-reminders.md) saved a verbatim reminder without a
yes, including in a tainted conversation, because waiting for a yes was
dropping ordinary reminders. The fired sentence was stored as a trusted
assistant message.

That broke two rules in `AGENTS.md`: a write outside the conversation
stages, and untrusted is sticky. A mail or web page read earlier in the
conversation could get the model to save a reminder whose text the page
chose. The person's own assistant would text it later. Once the page aged
out of the window, that text stayed in history as the assistant's own
words. A digest confirmed in a tainted conversation had the same second
half: its reports were stored as trusted.

## Decision

A verbatim reminder saves without a yes only when the conversation is
clean. In a tainted conversation it stages like any other job. Every job
row records whether the conversation that created it was tainted. What
that job later puts in history — the reminder text, a digest report, the
sender's "Sent to the other phone" line — carries that label.

## Consequences

- "Remind me" in a clean conversation is unchanged.
- Right after the assistant reads mail, the web, a list, or a relay, a
  reminder needs a yes.
- A repeating job created in a tainted conversation re-labels history each
  time it runs, so that conversation stays tainted while the job exists.
  Its proposals show the taint note until the job is cancelled.
- `jobs` gains `tainted`; `SCHEMA_VERSION` is 14. Existing rows default to
  clean, which is what they were treated as before.
- Spec 014's security review is superseded on this point by
  [spec 015](../specs/015-reminder-fixes.md). The PRD and architecture say
  so.
