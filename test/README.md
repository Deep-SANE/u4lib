# u4lib regression tests

From the u4lib root, after `make config`:

```sh
make test
```

Every invocation recompiles all five `test_*.cpp` executables into `bin/test/`
and runs them. `make -j2 test` permits parallel compilation; all executables
finish building before execution starts. The suite inherits the configured
compiler, null-space solver (dense QR by default) and Wigner backend (libwignernj) and
calls the existing
library functions. There is no test-specific solver selection.

Each case prints what is being tested and `[PASS]` or `[FAIL]`. Comparisons with
saved data say “against documented values”; no coefficient values are dumped,
even on failure. Failures identify the case, reference label or identity.
Each executable reports totals, followed by an overall result. A failed check,
exception, crashed executable or compilation error gives a nonzero make exit
status. Other executables still run after a test executable fails.

## Scope and label conventions

All external, intermediate and final **SU(4)** labels in the chosen cases obey
`0 <= P1,P2,P3 <= 5`. A triple means SU(4) Dynkin labels; its normalized U(4)
labels are `[P1+P2+P3,P2+P3,P3,0]`. Four labels in the tutorial are U(4) labels,
not Dynkin labels. Spin/isospin and magnetic labels are doubled integers:
`SS=2S`, `TT=2T`, `MMS=2MS`, `MMT=2MT`. All multiplicity indices are zero-based.

- `test_basis.cpp`: every one of the 216 Dynkin triples within the bound.
  Checks the Weyl dimension, Gelfand enumeration and ordering, spin-isospin
  branching dimension, conjugation twice, lookup, determinant shifts and
  forbidden spin/isospin channels. Also checks the fundamental decompositions.
- `test_canonical.cpp`: complete U(4)>U(3) reduced blocks for the couplings
  below, including every rho and eta. Checks normalization for every output
  U(3) branch, reconstructs full canonical CGs at the first, middle and last
  output Gelfand states, and checks scalar and singlet identities.
- `test_physical.cpp`: complete spin-isospin reduced blocks for the same
  couplings, including every allowed S/T and zeta/eta index. Checks
  orthogonality and independently reconstructs one complete zeta/eta block
  per coupling from canonical coefficients at low magnetic projections.
- `test_conjugation.cpp`: the 15 representative irreps listed below, with
  every allowed S,T,MS,MT. Checks documented Gelfand signs, physical
  conjugation, single/all-M consistency, orthogonality and both S_- and T_-
  lowering. Self-conjugate matrices are documented without assuming that zeta
  is preserved, and checked for symmetry, projection independence and square.
- `test_recoupling.cpp`: scalar U6/Z6/U9 identities plus complete channel
  matrices with outer multiplicities, including the tutorial cases. Every
  matrix entry is checked against a saved signed value; complete rows and
  columns are checked for orthogonality. Channel counts and block dimensions
  are checked too.

The coefficient cases are representative rather than exhaustive tensor
products of all 216 irreps. Physical transformations at the largest joint
labels are much more costly than basis enumeration; the expensive physical
checks use a fixed documented selection instead of silently skipping cases.

## Coupling cases

| A | B | C | Purpose |
|---|---|---|---|
| (0,0,0) | (1,2,1) | (1,2,1) | Scalar identity with branching multiplicity |
| (1,0,0) | (1,0,0) | (2,0,0), (0,1,0) | Fundamental 4 x 4 |
| (1,0,0) | (0,0,1) | (0,0,0), (1,0,1) | Singlet and adjoint |
| (1,0,1) | (1,1,1) | (1,1,1) | Tutorial canonical block, rho and eta |
| (1,1,1) | (1,1,1) | (1,2,1) | Tutorial physical block, zeta and eta |
| (1,0,1) | (1,0,1) | (1,0,1) | Adjoint outer multiplicity |
| (2,1,0) | (1,0,1) | (2,1,0) | Mixed non-self-conjugate irrep |
| (5,0,0) | (1,0,1) | (5,0,0) | P1 boundary |
| (0,0,5) | (1,0,1) | (0,0,5) | P3 boundary and conjugate branch |
| (0,5,0) | (1,0,1) | (0,5,0) | P2 boundary |

Exact tutorial blocks are included in the complete tables and receive extra
checks that their advertised multiplicities are actually nontrivial:

- Canonical `[2110] x [3210] -> [3210]`, U(3) rows `[210],[320],[320]`.
- Physical `(111),(SS,TT)=(2,2) x (111),(4,2) -> (121),(4,2)`.

Physical conjugation selection:
`(000), (100), (001), (010), (101), (111), (121), (210), (012), (202),
(500), (005), (050), (311), (113)`.

## Recoupling selection

- Tutorial: three adjoints `(1,0,1)` with total `(2,1,0)`; includes
  AB=`[3100]`, BC=`[2110]`, AC=`[2200]` and every other allowed channel.
- Additional U6/Z6: three adjoints with total adjoint, and
  `(1,1,1) x (1,0,1) x (1,0,1)` with total `(1,1,1)`.
- Tutorial U9: four adjoints to the scalar, including the full block in
  which all four pair intermediates are adjoints.
- Additional U9: four adjoints to the adjoint, with multiplicities on the
  pair-coupling and final-coupling axes.

## Tolerances and data

Documented coefficients use `abs(actual-expected) <= 1e-12 + 1e-10*abs(expected)`.
Accumulated matrix residuals use absolute Frobenius-norm tolerance `1e-9`;
the independent physical contraction uses absolute and relative `1e-10`.
Nonfinite values always fail. Tests compare signed entries in the library's
basis, without sign alignment or multiplicity rotations. Assertions remain
active even in release builds, and the explicit test checks do not use assert.

The configured solver's RREF, Gram-Schmidt and sign convention fix the basis.
Floating-point thresholds can still matter for ill-conditioned examples;
failures should be investigated rather than automatically accepted as new data.

The version-controlled `.dat` files and their label schemas, conventions and
provenance are described in [reference/README.md](reference/README.md).
`make test` never writes them. To calculate candidate values separately:

```sh
make reference-tool
mkdir -p /tmp/u4-reference-candidate
./bin/test/generate_references /tmp/u4-reference-candidate
```

Review the numerical candidates and mathematical checks before manually
replacing documented values. The generator uses the library, so saved values
are regression baselines, not an independent mathematical proof. The analytic
identities and independent contractions supplement those comparisons.

The root `make test` uses this folder; old `src/test/` diagnostics remain intact.
New `test_*.cpp` files are discovered automatically. The suite has no dependency
on the u3lib test folder; u3lib remains a normal library dependency of u4lib.
