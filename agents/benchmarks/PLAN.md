# Benchmark campaign plan (u3lib & u4lib)

*Shared benchmark framework across `u3lib` and `u4lib`.*

## 1. Objective

Quantify and evaluate the impact of repository-local agent skills on autonomous AI coding agents using a controlled experimental design:
- **Control condition:** Library without repository-local agent skills (documentation stripped of agent guidance).
- **Treatment condition:** Library equipped with the repository-local `agents/` skill catalog and agent guide.

## 2. Benchmark Tasks

The benchmark suite focuses on the three most mathematically and architecturally demanding tasks:

- **Task 1 (Python) — Three-irrep recoupling (`tasks/01-u6z6recoupling.md`)**:
  Implement `consumer/recouple.py` to evaluate $U$ and $Z$ Racah recoupling matrices with full channel discovery and matrix orthogonality verification.
- **Task 2 (C++) — Four-irrep 9-coefficient batch exporter (`tasks/02-u9recoupling.md`)**:
  Build `consumer/u9_batch` to read requests from stdin, evaluate complete 9-coefficient recoupling blocks with multiple outer multiplicities, and serialize them deterministically.
- **Task 3 (C++) — OpenMP batch service for canonical and physical coupling (`tasks/03-coupling-omp.md`)**:
  Build `consumer/coupling_batch` to evaluate mixed batches of canonical and physical coupling coefficients concurrently across multiple threads using OpenMP (`initialize_thread()` / `finalize_thread()`) with deterministic ordering.

## 3. Evaluation Protocol & Metrics

- **Harness & Isolation:** Each trial runs in an isolated disposable sandbox workspace with the library pre-configured.
- **Hidden Fixtures & Grader:** Test fixtures and the verification grader (`evaluator/grade.py`) are strictly withheld until the agent completes its run.
- **Primary Metrics:**
  - **Task Success:** All numerical, structural, and interface checks passed by `evaluator/grade.py`.
  - **Total Tokens:** Input, cache, and output tokens consumed to complete the task.
  - **Wall-clock Time & Cost:** Total execution duration and inference cost.

## 4. Run Matrix & Sample Size

- Matched pairs: For each task and model tested, run $N$ control trials and $N$ treatment trials with randomized interleaving.
- Statistical testing: Success rate comparison, Welch's $t$-test for token counts, and cost delta.

## 5. Directory Layout

- `tasks/` — Task specifications (`01-u6z6recoupling.md`, `02-u9recoupling.md`, `03-coupling-omp.md`, and `common.md`).
- `evaluator/` — Grader (`grade.py`), test fixture generator (`generate.py`), and precomputed reference fixtures (`fixtures-pilot.json`, `fixtures.json`).
- `runner/` — Generic sandbox experiment runner and execution harness.
