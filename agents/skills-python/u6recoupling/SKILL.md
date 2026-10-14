---
name: u6recoupling
description: Compute U-type three-irrep SU(4) recoupling blocks or complete multiplicity-resolved matrices (Python).
---

# u6recoupling

Read [shared runtime instructions](../runtime.md) before execution.

Routes: `(A B)AB C -> T` and `A (B C)BC -> T`. Compute the four `u4.Eta` counts (all positive), then call `get_U4U6` — note **T precedes C** in the argument order:
`get_U4U6(A, B, T, C, AB, eAB, eAB_C, BC, eBC, eA_BC)`.
Accessor: `block(iAB, iAB_C, iBC, iA_BC)`, each index bounded by its count.

```python
import u4lib as u4

u4.initialize()
try:
    # Example with non-trivial outer multiplicities (eta = 2 on all 4 channels):
    # Three adjoints (1,0,1) coupling to (1,0,1) through intermediate (1,0,1):
    A = B = C = T = u4.irrep(1, 0, 1)
    AB = BC = u4.irrep(1, 0, 1)

    eAB = u4.Eta(A, B, AB)      # (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
    eAB_C = u4.Eta(AB, C, T)    # (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
    eBC = u4.Eta(B, C, BC)      # (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
    eA_BC = u4.Eta(A, BC, T)    # (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
    assert eAB == 2 and eAB_C == 2 and eBC == 2 and eA_BC == 2

    # Getter call (note T precedes C):
    block = u4.u4u6db().get_U4U6(A, B, T, C, AB, eAB, eAB_C, BC, eBC, eA_BC)
    assert block.size() == 16
    assert abs(block(0, 0, 0, 0) - 0.6875) < 1e-10            # 11/16
    assert abs(block(0, 0, 0, 1) - 0.3247595264) < 1e-8       # 3*sqrt(3)/16
    assert abs(block(0, 0, 1, 1) - 0.0625) < 1e-10            # 1/16
    assert abs(block(1, 1, 1, 1) - (41.0 / 48.0)) < 1e-10     # 41/48
finally:
    u4.finalize()
```

For complete change-of-coupling matrix assembly:
1. Enumerate left route channels `(AB, eta_AB, eta_AB_C)` using `u4.decomposition(A, B)` and `u4.Eta(AB, C, T)`.
2. Enumerate right route channels `(BC, eta_BC, eta_A_BC)` using `u4.decomposition(B, C)` and `u4.Eta(A, BC, T)`.
3. Order channels lexicographically and check row/column orthogonality of the assembled matrix.
See [test/recoupling_cases.h](../../../test/recoupling_cases.h) for full matrix checks.
