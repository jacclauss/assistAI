# Decisions

Choices made after [prd.md](../prd.md) and [architecture.md](../architecture.md)
were written: an open question closed, an invariant loosened, a provider or
model swapped. One file each, `NNNN-short-name.md`, never edited after merge.
A later decision supersedes an earlier one by saying so.

Format:

```markdown
# NNNN — Title

Date: YYYY-MM-DD
Status: accepted | superseded by NNNN

## Context
What forced the choice.

## Decision
What we do now, in one or two sentences.

## Consequences
What gets easier, what gets harder, which docs and tests changed.
```

| # | Decision |
| --- | --- |
| [0001](0001-reminders-stage-when-tainted.md) | A reminder waits for a yes when the conversation is tainted |
| [0002](0002-daily-repeats-keep-local-time.md) | A daily repeat keeps its local time of day |
| [0003](0003-reworded-reminders-wait-for-yes.md) | A reworded reminder waits for a yes |
| [0004](0004-relays-after-an-outage.md) | A relay after a restart or an outage |
