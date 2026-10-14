---
name: cano-generators
description: Compute canonical U(4) Gelfand-state generator Eij matrix elements and the resulting states using u4lib.
---

# Canonical generator action

Read [shared runtime instructions](../runtime.md). Use the generator methods
on `u4::gelfand` (defined in `include/internal/gelfand/`) together with the
raising-action predicates in [U4basis.h](../../../include/U4basis.h). These are full canonical
matrix elements, not SU(2)xSU(2)-reduced generator matrix elements. They do not
need coefficient database initialization.

For a source g = (n14,n24,n34,n44; n13,n23,n33; n12,n22; n11), a generator acts
as `E |g> = sum_h <h|E|g> |h>`; the matrix element is the named method evaluated
on the **source** state, and the target `h` is the pattern the corresponding
predicate connects. The top row `[n14,n24,n34,n44]` never changes. Validate all
ten integers and interlacing before constructing the source.

## Generator families

U(4) has 15 generators; on Gelfand states the off-diagonal ones split by which
sub-row entry changes, giving the methods below (lowering partners `Eji` swap
the predicate's two arguments). Diagonal weights are `e11,e22,e33,e44` with
`sum_i e_ii` = total quanta.

| Generator | Predicate(s) in U4basis.h | Method(s) on g |
|---|---|---|
| E12 / E21 | `E12` | `e12` / `e21` |
| E23 / E32 | `E23_12`, `E23_22` | `e23_12,e23_22` / `e32_12,e32_22` |
| E13 / E31 | `E13_12`, `E13_22` | `e13_12,e13_22` / `e31_12,e31_22` |
| E34 / E43 | `E34_13`, `E34_23`, `E34_33` | `e34_13,e34_23,e34_33` / `e43_*` |
| E24 / E42 | `E24_13_12 … E24_33_22` (6) | `e24_13_12 … e24_33_22` / `e42_*` |
| E14 / E41 | `E14_13_12 … E14_33_22` (6) | `e14_13_12 … e14_33_22` / `e41_*` |

A raising predicate `Exx(g, h)` is true when `h` is the (unique) target reached
from `g`; reject invalid targets **before** evaluating a method, since boundary
states can make a branch denominator vanish. Keep the signs the library returns.
The predicates are free functions (`u4::E12`, `u4::E23_12`, …) in
[U4basis.h](../../../include/U4basis.h); identify the target by its label pattern
(the generator's effect on the sub-rows) and read the matrix element from the
source state's method.

## Executable example

```cpp
u4::gelfand g(2, 0, 0, 0, 2, 0, 0, 2, 0, 1);            // valid state of (2,0,0)
assert(g.check());
assert(std::abs(g.e12() - std::sqrt(2.0)) < 1e-12);     // <h|E12|g>, h has n11=2
u4::gelfand g2(2, 0, 0, 0, 2, 0, 0, 1, 0, 1);
assert(std::abs(g2.e23_12() - 1.0) < 1e-12);            // E23 raises n12
```

## Presentation and checks

Return each nonzero target pattern with its signed matrix element; an absent
target means zero action, not a missing result. For a matrix, enumerate
`u4::gelfandBasis(irrep)` (target rows, source columns) and check
`sum_i E_ii = C1 * I`, the transpose relation `E_ij^T = E_ji` in this real
basis, and the su(4) commutators `[E_ij,E_kl] = d(j,k)E_il - d(i,l)E_kj`. Do not
hide a failed identity by adjusting signs or discarding nonfinite values.
