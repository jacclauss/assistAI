# 0002 — A daily repeat keeps its local time of day

Date: 2026-10-30
Status: accepted

## Context

A repeating job stores its gap as a number of seconds, and the scheduler
moves the next run forward from when the last run finished. A daily job is
a gap of 86,400 seconds. A slow run or a retry pushes every later run
later, and across a daylight-saving change the job moves by an hour: an
8:00 AM reminder created in December runs at 9:00 AM from March 14, 2027.

PRD § Jobs says "The first run and the gap between repeats can be any
future value the clock can represent." Read as a fixed number of seconds,
that sentence requires the drift. The PRD also lists "every morning at 7"
as open engineering, so the household's intent is a time of day, not a
count of seconds.

[Spec 016](../specs/016-wall-clock-schedules.md) moves jobs to wall-clock
schedules in the household zone.

## Decision

A repeat measured in whole days keeps the local hour and minute the person
named. On the day the clocks change, the gap between two runs is 23 or 25
hours. A repeat measured in minutes keeps a fixed elapsed gap and is not
tied to the clock face.

## Consequences

- The PRD § Jobs sentence about the gap between repeats is amended in the
  same commit that implements the change (spec 016, commit 5). It will say
  that a whole-day gap follows the local clock and a minute gap is elapsed
  time.
- A daily or weekly job is stored as a number of days and a local time. A
  gap in minutes that adds up to whole days is refused, so the model uses
  days for those.
- A whole-day repeat needs a time the person named. "Every morning" with no
  hour gets "what time?" back, which reverses the 0.1.8 behaviour.
- A repeating job whose time falls in the spring-forward gap runs an hour
  later on that one day and returns to its time the next day. In the
  repeated hour it runs once, at the first occurrence.
- The scheduler advances from the slot that was due, not from when the run
  finished, so slow runs and retries no longer push later runs.
- `jobs` gains the zone, an anchor, the day count, and the local time;
  `SCHEMA_VERSION` is 15. Architecture describes the schedule, and the
  tests that pinned the elapsed step change with it.
