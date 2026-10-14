---
name: dim
description: Calculate SU(4) or U(4) irrep dimensions with u4lib.
---

# dim

Read [shared runtime instructions](../runtime.md) before execution. Python
examples below assume `import u4lib as u4` and the lifecycle described there.

Validate the requested label convention before constructing the irrep.
For SU(4), use `u4.irrep(P1,P2,P3).dim()`; for U(4), use
`u4.irrep(n14,n24,n34,n44).dim()`. Dimension is unchanged by a determinant
shift (uniform `n44` offset). Return an integer.

`dim()` is available on both `su4.irrep` and `u4.irrep`, and is
shift-invariant.

```python
assert u4.irrep(1, 0, 1).dim() == 15           # SU(4) adjoint
assert u4.su4.irrep(1, 0, 1).dim() == 15       # dim on su4.irrep too
assert u4.irrep(2, 1, 1, 0).dim() == 15        # U(4) labels, same adjoint
assert u4.irrep(1, 0, 0).dim() == 4            # fundamental
assert u4.irrep(2, 0, 0).dim() == 10
```

The closed form is
`(1+P1)(2+P1+P2)(3+P1+P2+P3)(1+P2)(2+P2+P3)(1+P3)/12`. For a requested table,
preserve input order and label each row. Do not confuse dimension with the
number of distinct (S,T) states or their multiplicities.
