---
name: cano-list
description: Enumerate the canonical U(4) Gelfand-Tsetlin basis of a U(4)/SU(4) irrep with u4lib, including each state's subgroup labels.
---

# Canonical (Gelfand) basis enumeration

Read [shared runtime instructions](../runtime.md). Use `u4.gelfandBasis` from
[U4basis.h](../../../include/U4basis.h), which returns the list of `u4.gelfand`
states spanning an irrep. This enumerates the basis; it does not act with a
generator (that is [cano-generators](../cano-generators/SKILL.md)) or transform
to the physical basis (that is [phys-to-cano](../phys-to-cano/SKILL.md)). These
getters do not require coefficient database initialization.

## What it returns

- `u4.gelfandBasis(irrep)` is the fixed-order list of all U(4) Gelfand patterns:
  the constant top row `[n14,n24,n34,n44]`, then the U(3) sub-row `[n13,n23,n33]`,
  the U(2) sub-row `[n12,n22]`, and `n11`. Enumeration order is the nested
  descending loops over `(n13,n23,n33)` then `(n12,n22)` then `n11`. Do not
  re-sort it: coupling and generator skills index states by this order.
- Each `gelfand` carries its subgroup labels `P1()/P2()/P3()`, `Lambdax2()` and
  `MLambdax2()` (twice the embedded SU(2) angular momentum and its projection),
  `check()` (betweenness), and the 44 generator matrix elements `e11()`..`e41_33_22()`.
- Build the irrep with three SU(4) labels `u4.irrep(P1,P2,P3)` or four U(4) labels
  `u4.irrep(n14,n24,n34,n44)`; validate `n14>=n24>=n34>=n44`. A nonzero `n44`
  determinant shift is kept; do not normalize it away silently.

## Enumerate and check

```python
import u4lib as u4
u4.initialize()
try:
    irr = u4.irrep(1, 0, 0)                 # fundamental, dim 4
    basis = u4.gelfandBasis(irr)
    assert len(basis) == irr.dim() == 4     # basis size == irrep dimension
    assert all(g.check() for g in basis)    # every pattern is valid
    # weights e11..e44 are the occupation of each U(4) row:
    assert [ (g.e11(),g.e22(),g.e33(),g.e44()) for g in basis ] == \
           [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
finally:
    u4.finalize()
```

## Presentation and checks

Return each state with its position index and subgroup labels, in library order.
The strongest invariant is `len(gelfandBasis(irrep)) == irrep.dim()`; also every
`check()` is true, and `sum_i e_ii` equals the total quanta (C1) on each state.
Report the fixed order as the library gives it; do not renumber to match another
enumeration convention.
