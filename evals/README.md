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
```

Grade behaviour, not phrasing. The best new cases come from a real Signal
exchange that went wrong: copy the message, fake the tool results it saw,
and assert what should have happened.
