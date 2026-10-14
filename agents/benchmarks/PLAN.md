# Benchmark campaign plan — second model + Task 6 (u3lib & u4lib)

*Shared campaign; an identical copy lives in `u3lib/agents/benchmarks/PLAN.md`.*
*Runner: `run_claude_host.py` (u3lib) / `run_claude_host_u4.py` (u4lib),
host-mode, per-condition frozen snapshots, grader introduced post-hoc.*

## 1. Objective

Strengthen the control-vs-treatment evidence that the repository-local agent
skills help, by (a) adding a **second Claude model**, (b) adding a **new
comprehensive Task 6**, and (c) raising the per-cell sample size for tighter
statistics — all within a modest, spread-out token budget.

## 2. What we keep (do NOT re-measure)

The current **Claude Sonnet 5.5** (`claude-sonnet-5-5`) results stand and are not
to be repeated:

- u3lib: Task 4 (Py) and Task 5 (C++), control+treatment, n=5 — `results/pilot-claude-t4-v6`, `pilot-claude-u9-v6`.
- u4lib: Task 4 and Task 5, control+treatment, n=5 — `results/pilot-claude-u4-t4-v6`, `pilot-claude-u4-u9-v6`.

These are the reworked-skills + zero-leak-control runs (treatment lower on tokens
in all four, 5/5 correctness everywhere; significant on u4 T5 −33% t=−4.0 and u3
T4 −41% t=−2.7). Reuse them as the Sonnet rows.

## 3. New Task 6 — canonical + physical coupling + OpenMP (C++)

Spec: `runner/tasks/06-coupling-mt.md`. One C++ program `consumer/coupling_batch`
that, in a single OpenMP parallel region (per-thread `initialize_thread()` /
`finalize_thread()`), evaluates **both** canonical coupling (`wigU3U2db` /
`wigU4U3db`) and physical coupling (`wigU3SO3db` / `wigU4SU2SU2db`) over a mixed
request batch, then serializes results deterministically in request order. It
combines task 2 (canonical), task 3 (physical), and the OpenMP threading model
into one flagship C++ task.

## 4. Second model — Claude Haiku 5.5 (`claude-haiku-5-5`)

Recommended because it is the **most informative and the most budget-friendly**
contrast: skills should help a smaller/weaker model *more* (bigger discovery gap,
more convention mistakes to prevent), so a strong Haiku treatment effect — or
control failing where treatment succeeds — would be a headline result. Haiku is
also ~3× cheaper per token than Sonnet, which is what makes a larger-n campaign
affordable.

- **Risk:** a weaker model may fail some control runs on the hard C++ tasks
  (an older Haiku thrashed on Task 5 earlier). That is an *informative* outcome,
  not a problem — record it honestly (success rate is a first-class metric).
- **Optional stretch** (only if budget/time allow): a third point with **Opus
  5.5** (`claude-opus-5-5`) on Task 5 + Task 6 to show the effect at the top tier.
  Opus is expensive; treat as out-of-scope for the core budget.

## 5. Run matrix and sample size

Target **n = 10 per condition** (up from 5) for all new cells — ~30% tighter
confidence intervals than n=5 at modest extra cost. One "cell" = one
(library, task, model) at control+treatment = 20 runs.

| Model | Task 4 (Py) | Task 5 (C++) | Task 6 (C++, new) |
|---|---|---|---|
| Sonnet 5.5 | keep n=5 (done) | keep n=5 (done) | **new, n=10** (both libs) |
| Haiku 5.5 | **new, n=10** | **new, n=10** | **new, n=10** (both libs) |

New cells (Tier 1):
- Sonnet × Task 6 × {u3,u4} = 2 cells = 40 runs
- Haiku × {Task 4, Task 5, Task 6} × {u3,u4} = 6 cells = 120 runs
- **Tier 1 total: 8 cells, 160 runs.**

Optional Tier 2 — **generally not recommended** (see the freeze caveat in §6):
topping Sonnet Task 4/Task 5 from n=5 to n=10 is only valid if the +5 runs use
the **byte-identical v6 snapshot**. The v6 snapshot stripped the README but
**not** the manual from control, whereas this campaign uses the zero-leak freeze
(manual stripped too) — so a re-freeze would produce a *different* control and
the +5 would not pool with the original 5. If you want n=10 Sonnet rows, reuse
the exact v6 snapshot (do not re-freeze); otherwise keep Sonnet T4/T5 at n=5 and
spend the budget on the new n=10 cells. Default: **skip the top-up.**

## 6. Budget and the "≤ half my Pro plan" constraint

Per-run economics observed so far (136 runs, ~\$55 API-equivalent, mean
\$0.40/run):

| Task | mean cost/run | mean tokens | mean time |
|---|---|---|---|
| Task 4 (Py) | \$0.34 | 582k | 77s |
| Task 5 (C++) | \$0.37 | 659k | 101s |
| Task 6 (C++, est.) | ~\$0.60 (Sonnet) | ~0.9M | ~150s |

Estimated campaign cost (API-equivalent):
- Sonnet Task 6: 40 × ~\$0.60 ≈ **\$24**
- Haiku Tasks 4/5/6: 120 × ~\$0.18 (Haiku ≈ ⅓ Sonnet \$/token) ≈ **\$22**
- **Tier 1 ≈ \$46**; optional Tier 2 ≈ +\$14.

**On the Pro-plan quota (important caveat):** the Pro plan is metered by
usage *windows* (rolling multi-hour + weekly caps), not a dollar budget, so an
exact "% of plan" cannot be computed from \$. The grounding argument instead:
we already ran **136 runs in bursts without exhausting the plan**; Tier 1 is a
similar total (~160 runs) but **spread thinly over 14 days**, which is far gentler
on the rolling windows. Keep it safe by (i) capping to **≈1 cell (20 runs,
~10–15M tokens) per active day**, (ii) leaving most days idle so the window
recovers, and (iii) monitoring usage and **hard-stopping at ~half** if any
usage warning appears. If the plan feels tight, drop to the Minimal set (§9).

**Freeze consistency (read before running).** Standardize the whole new campaign
on the **current zero-leak freeze** (control: no `agents/`, and README *and*
manual stripped). The kept Sonnet-v6 T4/T5 used an earlier freeze that stripped
the README but not the manual — a negligible difference for cross-model
comparison (the manual is rarely read), but note it; each model's own
control-vs-treatment remains internally valid. Re-running `freeze_inputs.py`
**overwrites** the snapshot, so v6's input snapshot is lost (its *results* are
saved); if exact v6 re-runs ever matter, copy the current snapshot aside first.

## 7. Two-week schedule (phased, one cell per active day)

- **Days 1–2 — setup (no model runs):** generate Task 6 fixtures and add the
  `task==6` grader branch (§8); build the Haiku runtime/snapshots (reuse the
  current frozen snapshots — only the `--model` changes); smoke-test 1 run of
  each new task×model to confirm the harness + build before committing budget.
- **Days 3–11 — core runs, ≈1 cell/day (20 runs):** interleave to de-risk drift,
  e.g. Haiku-u3-T5, Haiku-u4-T5, Sonnet-u3-T6, Sonnet-u4-T6, Haiku-u3-T6,
  Haiku-u4-T6, Haiku-u3-T4, Haiku-u4-T4. Run each library's two conditions as one
  seeded, randomized batch (control+treatment interleaved, no 3-in-a-row), as in
  the v5/v6 series.
- **Days 12–13 — optional Tier 2** (top up Sonnet T4/T5 to n=10) if under budget.
- **Day 14 — analysis + report:** fold results into
  `results/combined/summary.tex` as new model rows and a Task-6 section; keep the
  honest n and variance caveats.

## 8. Prerequisites (do before Task 6 / Haiku runs)

1. **Task 6 fixtures + grader.** Mirror tasks 4/5: pick a fixed Task-6 request
   batch (mixing `C`/`P`, including a multiplicity>1 case), compute reference
   values with the library (an independent `evaluator/generate.py`-style script),
   and add a `task==6` comparison branch to `evaluator/grade.py` (it already
   builds `consumer/` for task 6). Introduce fixtures only *after* the agent exits.
2. **Haiku runtime.** No snapshot change — the runner copies the same frozen
   `snapshots-v2/` (u3) / `snapshots/` (u4); only `--model claude-haiku-5-5`
   changes. Confirm the Haiku CLI model id is available first.
3. **Decision — teach OpenMP in the skills?** Task 6's threading is *not* covered
   by any current skill. Decide whether to add a short per-thread OpenMP note to
   the C++ `runtime.md` (the `initialize_thread()`-per-OMP-thread pattern) so the
   threading portion is skill-guided — this ties to the Python-parallelism task
   (release-plan §6). If left out, Task 6 measures the coupling skills with
   OpenMP as a shared (control=treatment) baseline difficulty; note which choice
   was made, since it changes what the Task-6 gap means.

## 9. Minimal viable set (if budget tightens)

Drop Haiku Task 4 (Python has the weakest, noisiest signal) and skip Tier 2.
Core becomes: Haiku {T5, T6} × 2 libs + Sonnet T6 × 2 libs = 6 cells, 120 runs,
~\$35 — still delivers the two highest-value questions: *does the effect
replicate on a second model?* and *does it hold on the new combined C++ task?*

## 10. Analysis and integrity

Report success rate, mean tokens ± std, cost, and a Welch t per cell; pool the
new n=10 cells and state significance honestly (n is still small — the aggregate
direction is the robust claim, not any single per-task number). Never fabricate:
on a harness/config bug, re-grade saved `consumer/` outputs rather than
re-running; flag anomalous runs (implausibly short/cheap) and outliers. The only
control/treatment difference must remain `agents/` + README + manual (freeze
asserts this); keep the cross-language skill lint clean.
