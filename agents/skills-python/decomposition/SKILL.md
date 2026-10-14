---
name: decomposition
description: Enumerate the full SU(4)/U(4) Kronecker product series A x B, listing every output irrep with its outer multiplicity, using u4lib (Python).
---

# Tensor-product (Clebsch-Gordan) decomposition

Read [shared runtime instructions](../runtime.md) before execution. `u4.decomposition(A, B)` takes
`u4.irrep(...)` and returns parallel lists: the output irreps `C` in `A x B` and
their outer multiplicities, reordering its inputs by dimension internally (which
does not change the set). For the multiplicity of a single fixed `C`, use
[outer-mult](../outer-mult/SKILL.md) (`u4.Eta(A,B,C)`), which equals the matching
entry. The algebra needs no coefficient database, but calling it inside an
initialized session is fine.

```python
import u4lib as u4

u4.initialize()
try:
    # Two adjoints (1,0,1) x (1,0,1) [15 x 15 = 225]:
    # Demonstrates non-trivial outer multiplicity (eta = 2 for C = (1,0,1)).
    A = B = u4.irrep(1, 0, 1)
    outputs, etamaxs = u4.decomposition(A, B)

    assert len(outputs) == 6
    total = sum(e * C.dim() for C, e in zip(outputs, etamaxs))
    # 84 + 45 + 45 + 20 + 2*15 + 1 = 225
    assert total == A.dim() * B.dim() == 225
    assert all(e == u4.Eta(A, B, C) for C, e in zip(outputs, etamaxs))
    assert u4.Eta(A, B, u4.irrep(1, 0, 1)) == 2
finally:
    u4.finalize()
```

## Ordering and checks

- The returned order is the generation order, **not** sorted by dimension or
  labels. Present every `(C, etamax)` pair with explicit labels, and state
  whether they are SU(4) `(P1,P2,P3)` or U(4) labels.
- `etamax` is a count (outer multiplicity); valid eta indices are `0..etamax-1`.
  A pair appears once with its full multiplicity, not repeated.
- Always report `sum_C etamax*dim(C) == dim(A)*dim(B)`; a mismatch means the
  enumeration is incomplete or an input was invalid. The set is symmetric under
  `A<->B`; the conjugate series is the irrep-conjugate set.
