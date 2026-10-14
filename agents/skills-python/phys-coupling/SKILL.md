---
name: phys-coupling
description: Compute SU(2)xSU(2)-reduced physical coupling coefficients with zeta and eta multiplicities (Python).
---

# phys-coupling

Read [shared runtime instructions](../runtime.md) before execution.

These are the physical spin-isospin reduced coupling coefficients
`(A, SA TA) x (B, SB TB) -> (C, SC TC)`, the SU(4) analogue of u3lib's
SO(3)-reduced `WigU3SO3`. **Spin/isospin labels are doubled: `SS=2S`, `TT=2T`.**
For each leg validate the irrep and that `zeta = Zeta(irrep, SS, TT) > 0`
(use [phys-list](../phys-list/SKILL.md) to enumerate valid `(S,T)`), require
`Eta(A,B,C) > 0`, and the SU(2) triangle conditions on both spin and isospin.

Prefer the structured `get_U4SU2SU2` accessor over guessing how the flat array
reshapes; axes are `(zetaA, zetaB, zetaC, eta)`.

```python
import u4lib as u4

u4.initialize()
try:
    # Coupling (1,0,1) S=1 T=1  x  (1,1,1) S=1 T=1  ->  (1,1,1) S=1 T=1:
    # Demonstrates both inner multiplicity (zetaB=2, zetaC=2) and outer multiplicity (eta=3).
    # Doubled labels: SS = 2*S = 2, TT = 2*T = 2 on all three legs.
    A = u4.irrep(1, 0, 1)
    B = u4.irrep(1, 1, 1)
    C = u4.irrep(1, 1, 1)

    eta = u4.Eta(A, B, C)     # 3
    assert eta == 3
    assert u4.Zeta(A, 2, 2) == 1
    assert u4.Zeta(B, 2, 2) == 2
    assert u4.Zeta(C, 2, 2) == 2

    blk = u4.wigU4SU2SU2db().get_U4SU2SU2(1, 0, 1,  2, 2,
                                          1, 1, 1,  2, 2,
                                          1, 1, 1,  2, 2)
    assert (blk.zetaAmax(), blk.zetaBmax(), blk.zetaCmax(), blk.etamax()) == (1, 2, 2, 3)

    # Verified coefficients: blk(zetaA, zetaB, zetaC, eta)
    assert abs(blk(0, 0, 0, 0) - 0.2434446675) < 1e-8
    assert abs(blk(0, 0, 1, 0) - (-0.2655760009)) < 1e-8
    assert abs(blk(0, 1, 1, 0) - (-0.4647580015)) < 1e-8
    assert abs(blk(0, 0, 0, 2) - 0.4123538702) < 1e-8
finally:
    u4.finalize()
```

Report reduced coefficients; full magnetic CGs additionally require the SU(2)
spin **and** isospin CG factors for the requested `(MS, MT)`. For orthogonality,
sum over the complete required channel set, not one block. See
[physical checks](../../../test/physical_cases.h).
