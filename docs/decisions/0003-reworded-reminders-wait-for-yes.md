# 0003 — A reworded reminder waits for a yes

Date: 2026-10-30
Status: accepted

## Context

[Decision 0001](0001-reminders-stage-when-tainted.md) lets a verbatim
reminder save without a yes when the conversation is clean. In that case the
only thing that decides whether the yes is skipped is the model's own
`verbatim` flag (`_reminder_saves_now` in `broker.py`). The architecture
audit filed this as B-1, medium after verification. Asked "every day at 7,
check my email and summarize it", the model can save a verbatim job whose
prompt is "No new mail.". That job then texts the sentence every day and
never opens the inbox, and nobody was asked. The sentence is shown at once
in the server's reply, but the person never had to look at it before it was
armed.

A reminder is the person's own sentence, texted back to them later. When
the saved sentence is mostly words the person did not use, it is no longer
clearly their sentence, and the yes is the check that is missing.

## Decision

A verbatim reminder saves without a yes only when the conversation is clean
and at least 60% of its saved sentence's content words appear in the
person's message that turn. Content words are three or more letters, not
stopwords, casefolded, with a trailing s, ed, or ing removed, ignoring
"don't", "forget", "remember", and "reminder". Otherwise it stages like any
other job. This extends decision 0001 and does not supersede it: a tainted
conversation still stages every reminder.

## Consequences

- "Remind me to take the bins out" in a clean conversation is unchanged. A
  reminder the model rewords, or a digest saved as a verbatim sentence,
  shows its text and waits for a yes.
- Some ordinary reminders that the model paraphrases heavily will need a
  yes. That is the cost decision 0001 noted, now limited to reworded
  sentences. The 60% threshold is tuned with the offline replay in
  [spec 017](../specs/017-job-output-contract.md) before enforcement.
- It ships log-only first. `broker.reminder_unmatched` records the overlap
  ratio, and nothing stages because of it, until enforcement is turned on.
- The PRD § Jobs sentence "Saying 'remind me' saves it with no yes, unless
  the conversation already holds mail, web, calendar, list, or relay text" is
  amended in the same commit that turns enforcement on. The architecture's
  § Jobs paragraph on reminders changes in that commit too.
- `tests/test_broker.py::test_a_reminder_whose_text_is_not_the_persons_is_logged`
  covers the log-only stage, and
  `tests/test_broker.py::test_a_reminder_whose_text_is_not_the_persons_stages`
  covers enforcement.
