---
name: review-pr
description: Runs an independent review of an AssistAI branch with fresh-context subagent reviewers, a cross-model verifier that tries to refute each finding, and an author disposition for every survivor. Use before marking a PR ready, when the user asks for a code review, or after pushing fixes to review findings.
---

# Review a PR

The author does not review its own work. The agent that wrote the change
orchestrates this review; it does not do the reviewing. Reviewers are fresh
subagents that see the requirement and the diff, never the author's
reasoning, plan, or summary of what it did.

Review finds what tests miss. It does not replace `make check`; run that
first and fix it before spending reviewers on the diff.

## 1. Gather the packet

The packet is the only context reviewers get. Do not add commentary.

- The base: `git merge-base origin/main HEAD`.
- The diff: `git diff <base>...HEAD`, plus `git diff --stat`.
- The requirement: the spec in `docs/specs/` the branch implements, with its
  acceptance criteria. For a bug fix with no spec, the issue or the user's
  request quoted verbatim.
- The rules: `AGENTS.md` and `.cursor/BUGBOT.md`.

If there is no spec and no quoted request, stop and ask for one. A reviewer
with no requirement can only check style.

## 2. Size the review to the risk

| Diff touches | Local lenses |
| --- | --- |
| Only `docs/`, `.cursor/`, comments, or tests | none; Bugbot alone |
| `src/` outside the high-risk paths | spec |
| `broker.py`, `staging.py`, `agents.py`, `gateway.py`, `scheduler.py`, `jobs.py`, `relay.py`, `secrets.py`, `store.py`, `signal/`, `inference/`, `config/`, `.github/`, a tool, a prompt, or a dependency | spec, security |

Bugbot is the correctness reviewer. It runs on GitHub once the PR is ready,
so a local correctness lens only repeats it. Bugbot does not read the spec
or the PRD; that is what the local lenses are for. Two or three independent
reviewers find most of what more would find. Do not add lenses to feel
thorough.

## 3. Run the lenses in parallel

Launch every lens in one message with the Task tool, `run_in_background:
false`. Each prompt is self-contained: the packet, the lens, the report
format below, and these lines verbatim:

> You are reviewing someone else's change. Do not edit, stage, commit, or
> push anything. Read changed files in full, not only the hunks. Every
> finding cites `path:line` and states a concrete failure: the input or
> sequence, what happens, and who is affected. If you cannot name one, it
> is not a finding. Report nothing about formatting or naming; CI covers it.

Lenses, all `subagent_type: generalPurpose`:

- **Spec**, on `claude-opus-5-5-medium`. For each acceptance criterion: is
  it met, and does a named test fail if it regresses? Anything the diff
  does that the spec puts out of scope? Any doc that now describes the old
  behaviour?
- **Security**, on `grok-4.7-high`. The prompt is the question list
  from [review-invariants](../review-invariants/SKILL.md), each answered
  with a finding or "clean".

The models were picked by rerunning each candidate on PR #2, whose findings
were already verified. Opus found the behaviour bugs on the spec lens that
Grok missed. As the security lens, Grok found three real issues to Sol's
two, including the relay confirmation stored as trusted. Composer missed
most findings and confirmed one that was false, so it does not review here.
Rerun that comparison before changing a model.

Report format for every lens:

```
- severity: blocking | concern | nit
  where: path:line
  failure: input or sequence → what happens → who is affected
  evidence: the code path, or a test that would fail
```

## 4. Verify before reporting

Merge the lens reports and drop exact duplicates. Then launch one verifier:
`subagent_type: generalPurpose` on `gpt-5.6-sol-medium`, a family different
from both lenses, so no model checks its own findings. Sol was not tried
as a verifier in the comparison; replace it if it confirms a finding the
code refutes. Give it
the packet and the merged findings, not which lens raised each one.

> For each finding, try to prove it wrong against the code on this branch.
> Trace the path; run `uv run pytest` on a narrow test, or write a
> throwaway test under `/tmp`, when that settles it. Do not edit the repo.
> Return each finding as `confirmed` (with the evidence that settled it),
> `refuted` (with the line that refutes it), or `unproven`. Merge findings
> that share a root cause. Re-rank severity: `blocking` means a broken
> invariant from AGENTS.md, data loss, or an acceptance criterion not met.

Drop `refuted`. Keep `unproven` only at `concern` or above, labelled as such.

## 5. Dispose of every finding

The author answers each surviving finding, and each Bugbot comment once the
PR is ready, in a table in the PR body under `## Review`:

| Finding | Severity | Disposition |
| --- | --- | --- |
| path:line, one line | blocking | fixed in `<sha>`, test `test_name` |
| path:line, one line | concern | declined: reason |

- A `blocking` finding is fixed, or the human explicitly accepts it. The
  author cannot decline one alone.
- A fix for a correctness or security finding comes with a test that failed
  before the fix.
- Declining a `concern` needs a reason a reviewer could check. "Out of
  scope" names where it will be handled.

## 6. Re-review the fixes

After fixes, run the verifier again on the new diff with the old findings.
Do not rerun the full lens set unless the fixes changed the design. Stop
after two rounds and hand the remainder to the human.

## 7. Hand off

Mark the PR ready only when no `blocking` finding is open. The human reads
the disposition table and the Bugbot comments on GitHub, and decides
whether to merge. Agents never approve or merge their own PRs.
