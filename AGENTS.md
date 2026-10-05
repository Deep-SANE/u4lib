# AGENTS.md — u4lib

Guidance for AI agents (and new contributors) working in **u4lib**: a C++
library for fast, accurate evaluation of **U(4) / SU(4) (Wigner supermultiplet)
coupling** and **Racah recoupling** coefficients.

- Paper: https://doi.org/10.1140/epjp/s13360-024-05581-6
- Manual: TBA (LaTeX source in `doc/`).
- **u4lib layers on u3lib** and follows the same conventions; **SAM** is the main
  consumer. Read the sibling `u3lib/AGENTS.md` too — structure, build system, and
  most gotchas are shared; this file notes only what is SU(4)-specific.

This repo is being prepared for public, AI-assisted release. A thin **MCP
server** wrapping the Python bindings (both u3lib and u4lib) is planned (see
"Public-release direction" at the end); keep the conventions and
input-validation notes below accurate, as that server will encode them.

## Build & test

Everything resolves from the **qcx-config environment** — no machine paths are
hardcoded. Source it first (from the QCX root), then use the make targets:

```bash
source "$QCX_ROOT/qcx-config/env.d/<you>.env.sh"
make          # build the C++ programs in src/ (-> bin/)
make test     # recompile AND run all regression tests (test/, -> bin/test/)
make python   # build the pybind11 extension -> python/u4lib.so
make clean
make reference-tool   # build the reference-data generator without running it
```

- `make test` is the regression gate (five `test_*.cpp`: basis, canonical
  SU(4)⊃U(3) CG, physical SU(2)×SU(2) CG, conjugation, recoupling). It rebuilds
  and runs, printing labeled `[PASS]`/`[FAIL]` without dumping coefficients; see
  `test/README.md` and `test/reference/README.md`. **Run it after any change to
  `src/` or `include/`.**
- `make python` is a **separate** target. The extension layers on the u4lib +
  u3lib + qcx-utils headers (header-only) and carries **no** MPI/SLEPc/sns/libsam
  — only libgomp + static wigxjpf. Details in `python/BUILD_NOTES.md`; read it
  before touching the bindings. `./build.sh` wraps `make -C python`.
- Dependencies (via qcx-config `make.d`): Boost headers, Eigen, **wigxjpf**,
  OpenMP — same as u3lib.

## Layout

Mirrors u3lib, with SU(4) headers in `include/`:

- irreps & basis: `U4irrep.h`, `U4gelfand.h`, `U4basis.h`, `U4decomposition.h`,
  `U4generatorsRME.h`, `U4SU2SU2_to_U4U3U2U1.h`.
- coupling: `U4U3xU4U3.h` (SU(4)⊃U(3) canonical reduced CG, `WigU4U3`),
  `U4SU2SU2xU4SU2SU2.h` (physical SU(2)×SU(2) = spin×isospin reduced CG,
  `WigU4SU2SU2`).
- recoupling: `U4_U6Z6Coeffs.h` (U-coefficient `U6`, Z-coefficient `Z6`),
  `U4_U9Coeffs.h` (9-symbol).
- `Definitions.h` — the structured containers (`U4U3`, `U4SU2SU2`, `U4U6`,
  `U4Z6`, `U4U9`); `u4lib.h` is the umbrella header.
- `src/`, `test/`, `python/`, `bin/` (build output — do not edit), `doc/`.

## Using the library — conventions and gotchas

Shared with u3lib (initialize/finalize discipline, DB-singleton-only access,
asserts-on → aborts, `U4U9` span lifetime). SU(4)-specific points:

- **Initialize cascades down.** `initialize()` sets up u4 → u3 → qcx_utils
  (Wigner backend + SO(3) caches + the U(3) **and** U(4) databases); `finalize()`
  tears down in reverse. `is_initialized()` reports state. So within u4lib the
  co-use cascade is automatic — but if you also drive **standalone u3lib** in the
  same process, still follow the shared-backend rule: init both up front,
  finalize both at the very end, and never finalize one mid-workflow (shared
  wigxjpf backend; see `python/BUILD_NOTES.md` "Using u3lib and u4lib together").
- **Irreps:** `irrep(P1, P2, P3)` (SU(4) labels) or `irrep(n1, n2, n3, n4)`
  (U(4) rows); `su4.irrep` is the base. `Eta(a, b, c)` is the outer coupling
  multiplicity of `a⊗b→c` (the SU(4) analogue of u3's `Rho`).
- **(S,T) branching uses `Zeta`, not SO(3)/`Kappa`.** `Zeta(P1,P2,P3,SS,TT)`
  (or `Zeta(irrep, SS, TT)`) is the branching multiplicity of the spin-isospin
  state `(S,T)` in the irrep, with **`SS=2S`, `TT=2T`** (doubled). It is valid
  only on the parity-correct `(S,T)` grid — off-grid it returns 0. **Enumerate
  valid states with `physicalBasis(irrep)`** rather than scanning `Zeta`.
- **Coefficients come only from the DB singletons:** `wigU4U3db()`,
  `wigU4SU2SU2db()`, `u4u6db()`, `u4z6db()`, `u4u9db()` → the `Definitions.h`
  containers. No free coefficient functions.
- **Asserts are ON (no `-DNDEBUG`).** A forbidden/empty coupling returns an empty
  container (safe to query for shape, **not** to index); indexing out of range
  `abort()`s the process. In Python, DB access before `initialize()` / after
  `finalize()` raises `RuntimeError`, but a bad index still aborts. Any
  public-facing wrapper (incl. the MCP server) must validate and raise, not abort.

### Getter signatures and multiplicity arguments

`η(a,b,c) = Eta(a,b,c)` is the outer multiplicity of `a⊗b→c`. The coupling
(`Wig*`) getters take **no** multiplicity inputs — output is indexed by the
multiplicities; the recoupling getters take the `η` maxima explicitly, in fixed
interleaved positions (same layout as u3lib's `Rho` versions).

- `wigU4U3db().get(` `n14a,n24a,n34a,n44a, n13a,n23a,n33a,` `…b, …c )` —
  SU(4)⊃U(3) canonical reduced CG; three irreps by U(4) rows `[n14,n24,n34,n44]`
  with their U(3) sublabels `[n13,n23,n33]`, `a⊗b→c`. Returns a 2-D numpy array
  over `(ρ, η)` (no multiplicity input).
- `wigU4SU2SU2db().get(` `P1a,P2a,P3a,SSa,TTa,` `…b, …c )` — physical
  SU(2)×SU(2) reduced CG, `SS=2S, TT=2T`. Returns a **flat** numpy array indexed
  by `(ζA,ζB,ζC,η)`. `get_U4SU2SU2(...)` takes the same args and returns the
  structured `U4SU2SU2`.
- `u4u6db().get_U4U6(` `A, B, ABC, C, AB,` `η(A,B,AB), η(AB,C,ABC),`
  `BC, η(B,C,BC), η(A,BC,ABC) )` — recouples `(A·B)_{AB}·C ↔ A·(B·C)_{BC}`,
  both → `ABC`. (Arg names: `etaABmax, etaAB_Cmax, etaBCmax, etaA_BCmax`.)
- `u4z6db().get_U4Z6(` `B, A, ABC, C, AB,` `η(A,B,AB), η(AB,C,ABC),`
  `AC, η(A,C,AC), η(AC,B,ABC) )` — recouples `(A·B)_{AB}·C ↔ (A·C)_{AC}·B`,
  both → `ABC`. **First two irrep args are `B, A`** (Z convention exchanges the
  first pair). (Arg names: `etaABmax, etaAB_Cmax, etaACmax, etaAC_Bmax`.)
- `u4u9db().get_U4U9(` `A, B, C, D, AB, CD, AC, BD, ABCD,`
  `η(A,C,AC), η(B,D,BD), η(AB,CD,ABCD),`
  `η(A,B,AB), η(C,D,CD), η(AC,BD,ABCD) )` — 9-symbol, recouples
  `(A·B)_{AB}·(C·D)_{CD} ↔ (A·C)_{AC}·(B·D)_{BD}`, both → `ABCD`. The **3rd**
  multiplicity is the `(AB)(CD)→ABCD` route, the **6th** is the `(AC)(BD)→ABCD`
  route — the pair most easily swapped (an internal `Eta(AB,CD,ABCD)` assertion
  catches it). (Arg names: `etaACmax, etaBDmax, etaABCDmax, etaABmax, etaCDmax,
  etaACBDmax`.)

## Conventions & correctness

Phase/normalization conventions follow the paper above; concrete documented
values and the generation procedure live in `test/reference/`
(`test/reference/README.md`). State the convention explicitly when exposing
coefficients to a new consumer — a wrong one yields silently-off numbers. The
regression suite checks dimensions, Gelfand structure, SU(2)×SU(2) branching,
canonical/physical CG orthonormality across `η` and multiplicities, and
conjugation identities; Python bindings are spot-checked, with full numerical
validation against reference tables the author's to confirm (see BUILD_NOTES).

## House rules for agents

- Source qcx-config before building; never hardcode toolchain/dependency paths.
- `bin/` and `*.dSYM/` are build output; don't commit or edit them (`.gitignore`
  covers the usual cases — check before `git add`).
- After changing `src/`/`include/`, run `make test`; after changing `python/`,
  rebuild with `make python` and import with the **same** interpreter used to
  build (ABI must match).
- Keep `README.adoc`, `python/BUILD_NOTES.md`, and this file in sync with the
  public API; they are the contract the release and the MCP server depend on.

## Public-release direction

Planned: a thin **MCP server** exposing u3lib and u4lib coupling + `U6`/`Z6`/
`U9` recoupling as tools, wrapping the existing pybind bindings. v1 (interactive
use) and the public v2 share the same tool bodies; the public delta is
productization (reproducible install incl. Linux, stable/versioned schemas,
docs, CI/tests) plus two things that must be right from the first public cut:
**documented, fixed coefficient conventions** and **input validation so bad
requests raise instead of aborting** (the asserts-on behavior above). Build v1
with those solid and it hardens into v2 without a rewrite.
