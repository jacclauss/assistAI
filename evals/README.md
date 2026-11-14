# Evals

Golden behaviour cases run against the real model, prompt, catalog, and
broker, with every tool handler replaced by the case's canned result. Nothing
reaches Gmail, iCloud, the web, or a phone. It does spend Fireworks tokens.

```bash
make eval                                  # all cases, 3 trials each, primary model
uv run python -m evals --tag security      # one tag
uv run python -m evals --case relay-draft --trials 5
uv run python -m evals --model glm_5p3     # a candidate from manifest.toml
```

Exit status is non-zero when any case misses its pass rate (default: every
trial). Run it before merging a change to `system_prompt_for`, a tool
description, a model pin, or `max_tool_rounds`.

`make check` does not call the model, but `tests/test_evals.py` parses every
case and checks the grader, so a broken case file fails CI.

## Writing a case

One TOML file per case in `cases/`. The file name is the id.

```toml
id = "short-name"            # must equal the file name
tags = ["mail", "regression"]
user = "What the person texts."
# tools = [...]              # default: first agent in config/assistai.example.toml
# pass_rate = 0.67           # only for genuinely stochastic behaviour; say why
# now = 2026-10-30T15:04:00  # with timezone: a stopped household clock (see below)
# timezone = "America/Chicago"

[[history]]                  # optional earlier messages
role = "user"                # user or assistant
untrusted = true             # e.g. a relay record or an earlier page
content = "…"

[fixtures]                   # tool name -> exact string the handler returns
mail_inbox = '''{"inbox_unread": 7, "messages": []}'''

[expect]
calls = ["mail_inbox"]       # requested at least once
no_calls = ["relay"]
staged = ["mail_file"]       # in the proposal the person would see
no_staged = ["relay"]
reply_contains = ["7"]       # case-insensitive
reply_excludes = ["Gmail API"]
max_tool_calls = 3

[expect.arguments.job_create]  # the model's arguments; see below
equals = { "at.hour" = 20, "at.minute" = 53 }
absent = ["delay"]
```

### A case on the household clock

`now` (a TOML local date-time, read on the household's wall) and
`timezone` go together. With them, `get_time` is the production handler
read at `now`, the system prompt's "Local time now" names the same moment,
and staging fixes a named `at` to a date on that clock, as the Pi does. A
`get_time` fixture beside `now` is an error. Without them, `get_time`
returns `{"ok": true}`, which passes the "get_time before at" gate and
names no time.

A case with a clock may start with stored jobs. `job_list` then returns
the production listing of these rows, so the times in it are the
formatter's words, and a `job_reschedule` or `job_cancel` finds them by
name or id (`j00000001` for the first row). A `job_list` fixture beside
`jobs` is an error.

```toml
[[jobs]]
name = "mgmplus cancel reminder"
prompt = "Cancel MGM+ before the rate goes up."
next_run = 2026-12-10T08:00:00   # local, like now
every_days = 1                   # omit for a one-time job
# verbatim = true                # the default
```

### Arguments

`[expect.arguments.<tool>]` grades the arguments of the first call of that
tool the broker staged or ran. A call it refused, such as an `at` sent
before `get_time`, is skipped, because the person never gets it. The
arguments are the model's own, not the copy staging stores: the question
is whether it copied the person's words, and the staged copy has already
turned "tomorrow" into a date. If no call of the tool was staged or run,
the trial fails.

Each table takes one or more of:

- `equals`: dotted path to value. Every path must hold that value.
  `"at.hour" = 8` and `at = { hour = 8 }` are the same expectation.
- `absent`: dotted paths that must not be set. JSON `null` counts as not
  set, as the parsers read it.
- `any_of`: tables of path to value. At least one table must match in
  full. Use it where two arguments mean the same thing, such as
  `any_of = [{ "at.day" = "tomorrow" }, { "at.date" = "2026-10-31" }]`.

Values are strings, numbers, or booleans. Strings compare without case,
numbers by value, and `true` is not `1`. An unknown key, an empty table,
a path both set and absent, or a list or date value fails the case file.

Grade behaviour, not phrasing. The best new cases come from a real Signal
exchange that went wrong: copy the message, fake the tool results it saw,
and assert what should have happened.
