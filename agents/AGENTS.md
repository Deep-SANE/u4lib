# u4lib calculation skills

Skills come in two parallel sets. **Choose the one matching the task's target
language:**

- **`skills-python/`** — for a Python calculation (the `u4lib` extension).
- **`skills-cpp/`** — for a C++ calculation (the `u4::` header API).

Both sets contain the same 16 skills under the same names; each skill carries the
conventions plus one worked example in its language, and each set has its own
`runtime.md` (environment, build, and initialize/finalize lifecycle). Pick the
folder first, read its `runtime.md`, then read the chosen skill's `SKILL.md`
(`skills-python/<name>/SKILL.md` or `skills-cpp/<name>/SKILL.md`) and execute.
Load additional skills only when the task needs them. Resolve each link relative
to the file containing it. Repository overview and developer build rules are in
[README.md](../README.md).

**First-time setup.** If the library is not built yet (no
`config/config.generated.mk`, or no `bin/` / `python/u4lib.so`), follow
[install/SKILL.md](install/SKILL.md) — `make config` then `make` — before using
any skill below.

| Skill | Python | C++ | Use for |
|---|---|---|---|
| dim | [skill](skills-python/dim/SKILL.md) | [skill](skills-cpp/dim/SKILL.md) | SU(4)/U(4) irrep dimensions. |
| casimirs | [skill](skills-python/casimirs/SKILL.md) | [skill](skills-cpp/casimirs/SKILL.md) | C1, C2, C3, C4 with explicit U(4) labels and SU(4) normalization. |
| cano-list | [skill](skills-python/cano-list/SKILL.md) | [skill](skills-cpp/cano-list/SKILL.md) | Canonical U(4) Gelfand basis of an irrep with its subgroup labels. |
| phys-list | [skill](skills-python/phys-list/SKILL.md) | [skill](skills-cpp/phys-list/SKILL.md) | Physical spin-isospin basis (zeta, S, T) of an irrep. |
| cano-generators | [skill](skills-python/cano-generators/SKILL.md) | [skill](skills-cpp/cano-generators/SKILL.md) | Canonical U(4) Eij matrix elements and resulting Gelfand states. |
| phys-generators | [skill](skills-python/phys-generators/SKILL.md) | [skill](skills-cpp/phys-generators/SKILL.md) | SU(2)xSU(2)-reduced matrix elements of the physical generators S, T, U. |
| phys-to-cano | [skill](skills-python/phys-to-cano/SKILL.md) | [skill](skills-cpp/phys-to-cano/SKILL.md) | Expand a physical state at specified (MS,MT) in the canonical Gelfand basis. |
| inner-mult | [skill](skills-python/inner-mult/SKILL.md) | [skill](skills-cpp/inner-mult/SKILL.md) | Spin-isospin branching multiplicity zeta of (S,T) in an SU(4) irrep. |
| outer-mult | [skill](skills-python/outer-mult/SKILL.md) | [skill](skills-cpp/outer-mult/SKILL.md) | Outer multiplicity eta for a specified tensor-product coupling. |
| decomposition | [skill](skills-python/decomposition/SKILL.md) | [skill](skills-cpp/decomposition/SKILL.md) | Full SU(4)/U(4) product series A x B with every output irrep and multiplicity. |
| cano-coupling | [skill](skills-python/cano-coupling/SKILL.md) | [skill](skills-cpp/cano-coupling/SKILL.md) | SU(4)>U(3) reduced canonical coupling coefficients over (rho, eta). |
| phys-coupling | [skill](skills-python/phys-coupling/SKILL.md) | [skill](skills-cpp/phys-coupling/SKILL.md) | SU(2)xSU(2)-reduced physical coupling coefficients with zeta and eta. |
| u6recoupling | [skill](skills-python/u6recoupling/SKILL.md) | [skill](skills-cpp/u6recoupling/SKILL.md) | U-type three-irrep recoupling blocks or complete matrices. |
| z6recoupling | [skill](skills-python/z6recoupling/SKILL.md) | [skill](skills-cpp/z6recoupling/SKILL.md) | Z-type three-irrep recoupling between AB-C and AC-B routes. |
| u9recoupling | [skill](skills-python/u9recoupling/SKILL.md) | [skill](skills-cpp/u9recoupling/SKILL.md) | Four-irrep 9-coefficient recoupling with all six multiplicity axes. |
| conjugation | [skill](skills-python/conjugation/SKILL.md) | [skill](skills-cpp/conjugation/SKILL.md) | Gelfand or physical-state conjugation, incl. self-conjugate zeta mixing. |

For composite requests, combine the relevant skills (from the same language set)
and preserve all channel labels. If a request leaves the coupling route, irrep
convention, or selected multiplicity ambiguous, resolve it before reporting a
single coefficient. These are repository-local instructions for code-executing
agents; no MCP server is required. They do not install themselves into any agent
application's skill registry. The user can explicitly point an agent to this
catalog.
