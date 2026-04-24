# Review instructions

These rules override the default review guidance. Treat them as the highest-priority
instruction block for any review of this repo. If something here contradicts a more
generic review habit, follow these.

## Severity levels

Every finding carries one of three priority labels:

- **P0** — blocks merge.
- **P1** — worth fixing, not blocking.
- **P2** — informational.

Write labels as plain text (`P0`, `P1`, `P2`) in finding headers. Do not use
emoji or colored markers. Use judgment on what belongs at which level — this
repo does not enumerate P0 cases; read the code and decide.

## Default bias: rigor

Reviews gate merges. This is an open-source repo that takes PRs from anyone; the
maintainer team is small and relies on the review to catch what they don't have
time to verify themselves. **Default bias is rigor, not speed.** When in doubt
on a P0-class concern, investigate further before deciding whether to flag — a
false negative ships a bug to production, a false positive costs the contributor
one round trip.

Rigor is not nitpicking. The P1 cap, "do not report" skip list, and verification
bar all still apply. Rigor means going deep on a small number of real concerns,
not surfacing a large number of shallow ones. Prefer one well-investigated P0
over three speculative P1s.

**Hold the line on P0.** If the author pushes back on a P0 finding without a fix
that actually addresses the root cause, re-state the concern with added
citations. Only accept the pushback if the author points to code or behavior you
missed. Do not soften a P0 because the contributor is polite or new to the repo.

For P1 and P2: if the author defers or pushes back without fixing, accept it
silently — do not re-flag on subsequent commits. P1/P2 are informational; the
author may defer to a follow-up issue at their discretion.

If Claude and the author repeatedly disagree on the same class of finding, the
signal is that REVIEW.md is missing a rule; note it once in the PR summary as
`suggest-rule: <short description>` and stop.

## Investigate before posting

The depth of your analysis determines the strength of your finding. For any
P0-class concern, before writing it up:

- Read the relevant callers and callees, not just the diff. Use Read and Grep
  to open files the diff doesn't touch but the changed code interacts with.
- Trace the full chain end-to-end for routing, auth, and agent-loop findings.
  Cite each hop by `file:line`, not just the suspicious line.
- Check whether the codebase already has an established pattern for this kind
  of change (`grep` for similar call sites, similar tool definitions, similar
  route guards). If the PR introduces a new approach where an established
  pattern exists, flag that — divergence from the existing pattern is usually a
  regression vector even when the new code "works."
- Confirm the specific behavior you're claiming. "This breaks X" must be
  grounded in either the code handling X or a test exercising X, not in
  inference from naming or structure.

A finding you "spotted" by scanning the diff is more likely to be a false
positive than a finding you verified by reading the code around it.

## P1 cap

Report at most **3** P1 findings per review. If you found more, say "plus N
similar items" in the summary. If everything you found is P1 or below, open the
summary with "No blocking issues."

## Re-review convergence

If this PR has already received a Claude review (there is a prior review comment
by the `claude` bot), suppress new P1 findings and post only P0 ones. Do not
re-post P1s that were already flagged on earlier commits. If the author pushed a
fix for a previously flagged issue, acknowledge it in one line rather than
re-flagging.

## Do not report

Anything in these paths — skip entirely:

- `frontend/node_modules/**`, `**/*.lock`, `uv.lock`, `package-lock.json`
- `hf_agent.egg-info/**`, `.ruff_cache/**`, `.pytest_cache/**`, `.venv/**`
- `session_logs/**`, `reports/**`
- Anything under a `gen/` or `generated/` path

Anything speculative — do not post:

- "This might be slow" without a concrete complexity claim tied to a specific
  input size
- "Consider adding a test" without naming the specific behavior that is
  untested and would regress silently
- Hypothetical race conditions without a concrete interleaving

## Dependency PRs

PRs whose diff is a lockfile bump (`uv.lock`, `package-lock.json`), a
`pyproject.toml` version change, or a new dependency need a different check
than code-behavior PRs. The code rules above mostly don't apply; the risks
shift to provenance, supply chain, and framing. For these:

- **Verify claimed CVEs.** If the PR body or title references a CVE ID, the
  CVE must resolve in the NVD (nvd.nist.gov) or the GitHub Advisory Database
  (github.com/advisories). If you cannot find the CVE, flag as P0 — fabricated
  CVE IDs are a known supply-chain attack pattern and merging lends them a
  false audit trail.
- **Title version must match the lockfile diff.** If the title says "upgrade X
  to 1.6.9" but the lockfile shows `1.5.0 → 1.7.0`, the PR is mislabeled or
  doing more than it claims. Mismatch is P0 regardless of whether the bump
  itself is safe — future maintainers grepping the commit history for the
  stated version will be misled.
- **Explain any new transitive deps.** If the lockfile bump pulls in a package
  that was not previously present, the PR body must name it and justify it.
  Unexplained new transitive deps — especially from authors with no prior
  contributions to the repo — are P0 supply-chain risk. Do not approve.
- **No code-behavior claims without code changes.** If a dep-only PR claims to
  "fix" a specific bug, "add" a feature, or "patch" a vulnerability that would
  require source changes to verify, the claim is false. Flag the framing as P0
  and note that the dep bump itself may still be fine in isolation.

## Verification bar

Every behavior claim in a finding must cite `file:line`. "This breaks X" is not
actionable without a line reference. If you cannot cite a line, do not post
the finding.

## Summary shape

Open the review body with a single-line tally and an explicit merge verdict, on
two lines:

```
2 P0, 3 P1
Verdict: changes requested
```

Valid verdicts:

- **Verdict: ready to merge** — no P0 findings, contributor can merge as-is
  once any CI passes
- **Verdict: changes requested** — at least one P0 that must be addressed
  before merging
- **Verdict: needs discussion** — a design-level concern the maintainer should
  weigh in on before the contributor iterates (use sparingly)

If it's a clean review, write `LGTM` followed by `Verdict: ready to merge`.

Then a **What I checked** bullet list — one line per major area you examined,
regardless of whether you found anything. This gives the maintainer visible
coverage at a glance and lets them decide whether to spot-check areas you
didn't touch.
