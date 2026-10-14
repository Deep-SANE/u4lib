---
name: u9recoupling
description: Compute four-irrep SU(4) 9-coefficient recoupling with all six multiplicity axes and correct result lifetime (Python).
---

# u9recoupling

Read [shared runtime instructions](../runtime.md) before execution.

Routes: `(A B)AB (C D)CD -> T` and `(A C)AC (B D)BD -> T`.
Calculate all six multiplicities before the call; all must be positive:

```python
import u4lib as u4

u4.initialize()
try:
    # Four adjoint irreps (1,0,1) coupling to total scalar (0,0,0) via (1,0,1):
    # Demonstrates non-trivial outer multiplicities (eta = 2 on four axes).
    A = B = C = D = u4.irrep(1, 0, 1)
    AB = CD = AC = BD = u4.irrep(1, 0, 1)
    T = u4.irrep(0, 0, 0)

    eAB, eCD = u4.Eta(A, B, AB), u4.Eta(C, D, CD)
    eAC, eBD = u4.Eta(A, C, AC), u4.Eta(B, D, BD)
    eLeft, eRight = u4.Eta(AB, CD, T), u4.Eta(AC, BD, T)
    assert eAB == 2 and eCD == 2 and eAC == 2 and eBD == 2
    assert eLeft == 1 and eRight == 1

    # Getter arguments: (A, B, C, D, AB, CD, AC, BD, T, eAC, eBD, eLeft, eAB, eCD, eRight)
    block = u4.u4u9db().get_U4U9(A, B, C, D, AB, CD, AC, BD, T,
                                 eAC, eBD, eLeft, eAB, eCD, eRight)
    assert block.size() == 16

    # Verified coefficients: block(iAC, iBD, iLeft, iAB, iCD, iRight)
    assert abs(block(0, 0, 0, 0, 0, 0) - (-0.0625)) < 1e-10      # -1/16
    assert abs(block(0, 1, 0, 1, 0, 0) - 0.8125) < 1e-10         # 13/16
    assert abs(block(0, 1, 0, 0, 1, 0) - (-0.1875)) < 1e-10      # -3/16
    assert abs(block(1, 1, 0, 1, 1, 0) - (5.0 / 48.0)) < 1e-10   # 5/48
finally:
    u4.finalize()
```

Accessor: `block(iAC, iBD, iLeft, iAB, iCD, iRight)`.
* **Important Axis Ordering:** The 3rd multiplicity belongs to `AB x CD -> T`; the 6th to `AC x BD -> T`. Keep these distinct, especially when their sizes differ; an internal assertion catches any swap.

`u4u9db().get_U4U9(...)` returns an owning result in Python that you can keep and index across later DB calls; the returned object already owns its coefficients.
For a complete recoupling matrix:

1. Discover left-route pairs `(AB, CD)` using `u4.decomposition(A, B)` and
   `u4.decomposition(C, D)`; retain pairs with `u4.Eta(AB, CD, T) > 0`.
   Enumerate every `(iAB, iCD, iLeft)` for each retained pair.
2. Discover right-route pairs `(AC, BD)` using `u4.decomposition(A, C)` and
   `u4.decomposition(B, D)`; retain pairs with `u4.Eta(AC, BD, T) > 0`.
   Enumerate every `(iAC, iBD, iRight)` for each retained pair.
3. Label matrix rows by left-route channels and columns by right-route channels.
   Fill each entry from `block(iAC, iBD, iLeft, iAB, iCD, iRight)` for the
   corresponding four intermediate irreps. Preserve the library multiplicity basis.
4. Check row and column orthogonality only after assembling both complete channel
   sets. A block for fixed intermediate irreps is generally not orthogonal by itself.
See [test/recoupling_cases.h](../../../test/recoupling_cases.h) for full matrix checks.
