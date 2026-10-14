# Documented U(4) coefficient and phase values

Files are whitespace-delimited `label value` tables with 17 significant digits.
Comments start with `#`. The trailing colon belongs to each label. Duplicate,
missing, unused and nonfinite entries cause test failures. All eta, rho and zeta
indices are **zero-based**. Irrep triples mean `(P1,P2,P3)` and bracketed U(3)
rows mean `[n13,n23,n33]`; labels are explicit rather than implicit row offsets.

## Schemas

- `canonical.dat`:
  `A x B -> C : [a13,a23,a33] : [b13,b23,b33] : [c13,c23,c33] : rho : eta :`
  (no spaces in the actual label). These are U(4)>U(3) reduced coefficients,
  indexed by SU(3) coupling multiplicity rho and SU(4) multiplicity eta.
- `physical.dat`:
  `A x B -> C : SSA : TTA : SSB : TTB : SSC : TTC : zetaA : zetaB : zetaC : eta :`.
  These are SU(2)_S x SU(2)_T reduced coefficients; S/T labels are doubled.
- `conjugation.dat`:
  `irrep:GT:n13:n23:n33:n12:n22:n11:` stores the Gelfand conjugation sign.
  `irrep:phase:SS:TT:` stores the distinct-irrep physical sign at MS=S, MT=T.
  `irrep:D:SS:TT:zeta_target:zeta_source:` stores a self-conjugate matrix.
- `recoupling.dat`:
  `U6multi:A:B:C:total:AB:etaAB:etaAB_C:BC:etaBC:etaA_BC:`;
  `Z6multi:A:B:C:total:AB:etaAB:etaAB_C:AC:etaAC:etaAC_B:`;
  `U9multi:(1,0,1):total:AB:CD:etaAB:etaCD:etaAB_CD:AC:BD:etaAC:etaBD:etaAC_BD:`.
  U9 has four external adjoints. Labels give left then right channels, while
  the structured C++ accessor uses the axis order
  `(etaAC,etaBD,etaAB_CD,etaAB,etaCD,etaAC_BD)`.
  `U6dimension:A:B:C:total:`, `Z6dimension:A:B:C:total:` and
  `U9dimension:(1,0,1):total:` document complete channel dimensions.
  `scalar:U6:`, `scalar:Z6:`, `scalar:U9:` are analytic +1 identities.

## Complete recoupling matrices

| Coefficients | External irreps | Total | Dimension |
|---|---|---|---|
| U6/Z6 | three (1,0,1) | (2,1,0) | 6 x 6 |
| U6/Z6 | three (1,0,1) | (1,0,1) | 9 x 9 |
| U6/Z6 | (1,1,1), (1,0,1), (1,0,1) | (1,1,1) | 18 x 18 |
| U9 | four (1,0,1) | (0,0,0) | 9 x 9 |
| U9 | four (1,0,1) | (1,0,1) | 43 x 43 |

The first and fourth rows contain the exact tutorial blocks. The full matrices
also include every other allowed intermediate irrep and multiplicity index;
orthogonality is never asserted for an incomplete isolated block.

## Mathematical checks

A scalar coupled on the left gives identity coefficients. Full canonical
coefficients for R x conjugate(R) -> scalar are
`delta(gB,conjugate(gA))*conjugate_phase(gA)/sqrt(dim(R))` in this convention.
The tests restore the lower U(3)>U(2)>U(1) factors before checking this formula.

Conjugation is defined by the existing library map:

```
C|g> = (-1)^(n14+n34-n13-n23-n33-n12-n22-n11) |g.conjugate()>.
C^2 = (-1)^(P1+P3).
```

For P1 != P3, set `N=P1+P2+P3` and
`delta=P1+P3` if P1<P3, otherwise `delta=0`. The tested physical action is

```
C |(P1,P2,P3) zeta S MS T MT>
  = (-1)^(N-S-T+MS+MT+delta)
      |(P3,P2,P1) zeta S,-MS T,-MT>.
```

The reverse-direction sign is essential when P1+P3 is odd. For P1=P3,

```
C |irrep zeta S MS T MT>
  = (-1)^(S-MS+T-MT) sum_zeta' D[zeta',zeta]
      |irrep zeta' S,-MS T,-MT>.
```

D is obtained in the library's real basis at MS=S,MT=T. It is tested at every
projection, including reconstruction, symmetry and D^2=I. No diagonality is
assumed. An antilinear interpretation additionally conjugates coefficients of
superpositions; basis coefficients here are real.

Full coefficient orthogonality, channel dimensions and recoupling matrix row
and column orthogonality supplement the saved values. The physical test also
performs an explicit canonical contraction at low magnetic projections.

## Provenance and regeneration
Generated separately with `test/tools/generate_references.cpp`, using the
existing u4lib functions. Numerical baselines are not claimed to be independently
derived exact coefficients.

- Date: 2026-09-29.
- Platform: macOS arm64; GNU C++ 16.2.0.
- Dependencies: `u3lib` (supplying Eigen3, Boost, fmt, and libwignernj backend);
  dense QR null-space solver (`NULLSPACE_SOLVER=qr`).
- Basis fixing: library RREF, Gram-Schmidt and leading-nonzero sign convention.
- Tutorial source: `python/tutorial.ipynb`, canonical/physical and recoupling
  examples, supplemented by the cases listed in `test/README.md`.

From the u4lib root after `make config`:

```sh
make reference-tool
mkdir -p /tmp/u4-reference-candidate
./bin/test/generate_references /tmp/u4-reference-candidate
```

The tool runs mathematical checks while producing candidate tables. Review
those results before copying candidates into this directory. `make test`
reads these files and cannot regenerate them. Reference changes must be
intentional and documented rather than a way to make an unexplained failure pass.

## Validation

The separate corrected reference calculation passed all 50 coefficient,
conjugation and recoupling cases and reproduced all 7,100 saved values exactly
on this machine. The normal build-and-run suite passed all 271 labeled checks
across five executables, including exhaustive basis checks over the 216 allowed
Dynkin triples. An altered reference in an isolated temporary directory caused
a failing comparison and nonzero executable exit status. A subsequent make
dry run confirmed that every test executable is scheduled for recompilation.
