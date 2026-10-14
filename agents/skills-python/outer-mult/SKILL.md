---
name: outer-mult
description: Calculate outer multiplicity eta for a specified SU(4) tensor-product coupling.
---

# outer-mult

Read [shared runtime instructions](../runtime.md) before execution. Python
examples below assume `import u4lib as u4` and the lifecycle described there.

Construct A,B,C as `u4.irrep(P1,P2,P3)` and call `u4.Eta(A,B,C)` — the outer
multiplicity of `A⊗B→C`, the SU(4) analogue of u3lib's `Rho`. The nine-integer
overload is `u4.Eta(P1a,P2a,P3a, P1b,P2b,P3b, P1c,P2c,P3c)`. A zero result
denotes a forbidden output irrep, not an error for valid labels.

```python
import u4lib as u4
u4.initialize()
try:
    assert u4.Eta(u4.irrep(1,0,0), u4.irrep(1,0,0), u4.irrep(2,0,0)) == 1
    adj = u4.irrep(1, 0, 1)
    assert u4.Eta(adj, adj, adj) == 2       # adjoint^3 carries multiplicity 2
finally:
    u4.finalize()
```

Return the count separately from valid zero-based eta indices. For a full
product decomposition, use [decomposition](../decomposition/SKILL.md) rather
than scanning candidate C by hand; do not cap intermediate labels at a small
bound just because regression tests use one. Check
`sum_C eta*dim(C) = dim(A)*dim(B)` only when the decomposition is complete.
This skill alone does not produce coupling coefficients.
