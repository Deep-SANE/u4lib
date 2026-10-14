---
name: z6recoupling
description: Compute Z-type three-irrep SU(4) recoupling between AB-C and AC-B coupling routes (Python).
---

# z6recoupling

Read [shared runtime instructions](../runtime.md) before execution.

Routes: `(A B)AB C -> T` and `(A C)AC B -> T`. Compute the four `u4.Eta` counts (all positive), then call `get_U4Z6` — the getter **starts with B, A** (exchanging the first pair):
`get_U4Z6(B, A, T, C, AB, eAB, eAB_C, AC, eAC, eAC_B)`.
Accessor: `block(iAB, iAB_C, iAC, iAC_B)`. Note that container shape getters `rhoBCmax()`/`rhoA_BCmax()` refer here to the AC and AC_B channels.

```python
import u4lib as u4

u4.initialize()
try:
    # Example with non-trivial outer multiplicities (eta = 2 on all 4 channels):
    # Three adjoints (1,0,1) coupling to (1,0,1) through intermediate (1,0,1):
    A = B = C = T = u4.irrep(1, 0, 1)
    AB = AC = u4.irrep(1, 0, 1)

    eAB = u4.Eta(A, B, AB)      # (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
    eAB_C = u4.Eta(AB, C, T)    # (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
    eAC = u4.Eta(A, C, AC)      # (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
    eAC_B = u4.Eta(AC, B, T)    # (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
    assert eAB == 2 and eAB_C == 2 and eAC == 2 and eAC_B == 2

    # Getter call (note leading B, A, and T precedes C):
    block = u4.u4z6db().get_U4Z6(B, A, T, C, AB, eAB, eAB_C, AC, eAC, eAC_B)
    assert block.size() == 16
    assert abs(block(0, 0, 0, 0) - (-0.0625)) < 1e-10         # -1/16
    assert abs(block(0, 1, 1, 0) - 0.8125) < 1e-10            # 13/16
    assert abs(block(0, 0, 1, 1) - (-0.1875)) < 1e-10         # -3/16
    assert abs(block(1, 1, 1, 1) - (5.0 / 48.0)) < 1e-10      # 5/48
finally:
    u4.finalize()
```

For complete change-of-coupling matrix assembly:
1. Enumerate left route channels `(AB, eta_AB, eta_AB_C)` using `u4.decomposition(A, B)` and `u4.Eta(AB, C, T)`.
2. Enumerate right route channels `(AC, eta_AC, eta_AC_B)` using `u4.decomposition(A, C)` and `u4.Eta(AC, B, T)`.
3. Order channels lexicographically and check row/column orthogonality of the assembled matrix.
See [test/recoupling_cases.h](../../../test/recoupling_cases.h) for full matrix checks.
