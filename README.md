# u4lib

**u4lib** is an **AI-ready computational framework for U(4)/SU(4) algebra**
(the Wigner supermultiplet symmetry): irreps and their invariants, canonical and
physical bases, generators and their matrix elements, basis transformations,
tensor-product decompositions, and the **Wigner coupling** and **Racah
recoupling** coefficients — exposed through a C++ API, Python bindings, and an
agent-driven skill layer over all of it.

- **Paper:** [Eur. Phys. J. Plus **139**, 933 (2024)](https://doi.org/10.1140/epjp/s13360-024-05581-6)
- **Manual:** [Manual (PDF)](doc/Manual.pdf); LaTeX source in `doc/Manual.tex`.
- **Prerequisite / sibling library:** `u3lib` (SU(3)); **SAM** is the main consumer.

> **Note for AI Agents:** Do not parse this human documentation. Refer directly to [agents/AGENTS.md](agents/AGENTS.md) for repository guidance, first-time build/installation, runtime lifecycle, and task-specific skills.

---

## Scope

The framework covers the SU(4)/supermultiplet representation theory needed for symmetry-adapted structure calculations:

- **Irreps & invariants** — SU(4) $(P_1,P_2,P_3)$ and U(4) $[n_{14},n_{24},n_{34},n_{44}]$ labels, dimension, Casimir invariants $C_1, C_2, C_3, C_4$, and conjugation.
- **Bases** — canonical Gelfand–Tsetlin basis and physical $\mathrm{SU(2)}\times\mathrm{SU(2)}$ spin-isospin basis $(\zeta, S, T)$, with inner and outer multiplicities.
- **Generators** — canonical $E_{ij}$ matrix elements on Gelfand states, and $\mathrm{SU(2)}\times\mathrm{SU(2)}$-reduced matrix elements of physical $\mathbf{S}$, $\mathbf{T}$, and Gamow–Teller tensor $\mathbf{U}$.
- **Transformations** — physical $|(P_1,P_2,P_3)\,\zeta\,S\,M_S\,T\,M_T\rangle \leftrightarrow$ canonical Gelfand expansion.
- **Decomposition** — full Kronecker product series $A \otimes B$ with outer multiplicities.
- **Coupling & recoupling** — $\mathrm{SU(4)}\supset\mathrm{U(3)}$ canonical and $\mathrm{SU(2)}\times\mathrm{SU(2)}$ physical reduced Wigner coefficients, and $U$-, $Z$-, and $9$-coefficient Racah recoupling.

---

## Build & Test

u4lib layers on `u3lib` and needs no manual environment variables. Configure once, then build:

```bash
make config   # resolve dependencies and u3lib, writing config/config.generated.mk
make          # build C++ driver programs in src/ (-> bin/)
make test     # recompile and run regression test suite (test/, -> bin/test/)
make python   # build pybind11 extension -> python/u4lib.so (auto-detects a pybind11+Jupyter python; override PYTHON=...)
make clean
```

`make config` detects `u3lib` (checking `../u3lib`, `$U3LIB_DIR`, etc.) and inherits its toolchain, Eigen3, Boost, fmt, and libwignernj backend. It is interactive by default; for scripted use pass flags through, e.g. `make config CONFIG_ARGS="--yes"`.

**Installing with an AI agent.** Point a code-executing assistant at this repository and prompt, for example:

> Install u4lib in this repository: follow `agents/install/SKILL.md` — run `make config` (pointing to u3lib), then `make` and `make test` — and report the test result.

The agent resolves the steps through the repository-local [install skill](agents/install/SKILL.md).

- **Regression Tests (`make test`):** Tests dimensions, Gelfand structure, $\mathrm{SU(2)}\times\mathrm{SU(2)}$ branching, coupling orthonormality across $\zeta$ and $\eta$, and conjugation identities. Checks remain enabled in release builds (`-UNDEBUG`).
- **Python Extension (`make python`):** Builds `python/u4lib.so`. The interpreter is auto-detected (one with `pybind11`, preferring one that also has `ipykernel` for the tutorial); override with `make python PYTHON=/path/to/python`. See `python/tutorial.ipynb` for interactive usage.
- **Dependencies:** a C++20 compiler and OpenMP are required on the system; `u3lib` is the prerequisite and supplies Eigen3, Boost, fmt, and libwignernj. The null-space solver defaults to **KraNS** (`NULLSPACE_SOLVER=krans`), inherited from u3lib, with an automatic dense-QR fallback for small systems. Full configuration options are detailed in [doc/Manual.pdf](doc/Manual.pdf).

---

## Repository Layout

- `include/` — header-only C++ library (`u4lib.h` umbrella; irreps, bases, generators, coupling, recoupling, containers).
- `utils/` — forwarding headers routing utility includes to `u3lib/utils/`.
- `src/` — C++ example driver programs; compiled into `bin/`.
- `python/` — pybind11 bindings (`pyu4lib.cpp`) and tutorial notebook (`tutorial.ipynb`).
- `test/` — regression suite (`test_*.cpp`, `*_cases.h`, and reference tables under `reference/`).
- `doc/` — full reference manual (`Manual.pdf` and LaTeX source).
- `agents/` — AI-agent skill catalog (`AGENTS.md`), runtime instructions, and task skills.
- `bin/` — compiled binary output (not source; git-ignored).

---

## Documentation & AI Agent Skills

- **Full Reference Manual:** For exhaustive mathematical derivations, phase conventions, getter signatures, and solver backends, consult [doc/Manual.pdf](doc/Manual.pdf).
- **AI-Ready Calculations:** `u4lib` provides a repository-local skill package under [`agents/`](agents/AGENTS.md) that enables code-executing AI assistants to plan and drive calculations through natural-language prompts. A complete table of representative prompts with checked reference values is provided in **Table II of the [Manual](doc/Manual.pdf)** as well as in [`agents/AGENTS.md`](agents/AGENTS.md).
