# u4lib agent-skill benchmark — setup (scaffolding)

Mirrors `u3lib/agents/benchmarks` for the SU(4) library. **Design and tasks are
drafted; it is not yet runnable** — the remaining pieces are listed below.

## Design
- **Model:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), matching the u3lib Sonnet pilots.
- **Conditions:** `control` = u4lib with no agent materials (README stripped of
  agent references, no `u4lib/agents/`); `treatment` = u4lib with the `agents/`
  skill package (16 skills) + the README Agent-assisted section. Everything else
  identical. (u4lib already ships both, so control is produced by removing them.)
- **Tasks** (`runner/tasks/`, SU(4) analogues of the u3lib tasks): `01-basis`
  (C++), `02-canonical` (Py, wigU4U3db), `03-physical` (Py, wigU4SU2SU2db),
  `04-recoupling` (Py, u4u6db/u4z6db), `05-u9` (C++, u4u9db). Shared preamble in
  `runner/common.md`.
- **Recommended first:** `05-u9` then `04-recoupling` — the hardest,
  convention-heavy tasks, where the u3lib pilots showed the skills help.
- **Metrics / protocol:** identical to the u3lib pilot — binary
  success-within-1800s + total tokens; `claude -p` non-interactive, pinned model,
  isolated disposable workspace, hidden fixtures introduced only after the agent
  exits.

## Remaining to make it runnable (build checklist)
1. **Frozen workspaces** `runner/snapshots/{control,treatment}/workspace`: each =
   `common.md` + `environment.sh` + empty `consumer/` + the u4lib library **and
   the u3lib it depends on** + qcx-utils. treatment keeps `u4lib/agents/` and the
   README agent refs; control removes both. Add a `manifest.json` of file hashes.
2. **Host runtime**: a dir holding **both** `u4lib.so` and `u3lib.so` built for the
   miniforge interpreter (u4 `initialize()` cascades into u3), plus a host
   `environment.sh` pointing at the host toolchain + that dir on `PYTHONPATH`.
   (`u4lib/python/u4lib.so` is already built; verify it imports with `u3lib.so`
   alongside under the same interpreter.)
3. **Grader** `evaluator/grade.py` for the SU(4) tasks + `evaluator/fixtures-pilot.json`
   of precomputed expected values (generate on the host with u4lib, as u3lib's
   `evaluator/generate.py` does).
4. **Runner** `runner/run_claude_host_u4.py`: a copy of
   `u3lib/agents/benchmarks/runner/run_claude_host.py` with paths pointed at u4lib
   (SNAP, U4LIB_DIR, the two-extension runtime) and `MODEL=claude-sonnet-5-5`.
5. **Schedule** `results/pilot-claude-u4-<task>/schedule.json` (seeded, balanced).

Reuse the u3lib harness verbatim where possible; only the library paths, the
two-extension runtime, the SU(4) grader/fixtures, and the tasks differ.
