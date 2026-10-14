---
name: inner-mult
description: Calculate the SU(2)xSU(2) spin-isospin branching multiplicity zeta of (S,T) in an SU(4) irrep with u4lib.
---

# inner-mult

Read [shared runtime instructions](../runtime.md) before execution. Python
examples below assume `import u4lib as u4` and the lifecycle described there.

Here inner multiplicity means SU(4) > SU(2)xSU(2) (spin x isospin) branching:
the multiplicity `zeta` of a spin-isospin state `(S,T)` in the irrep. It is the
SU(4) analogue of u3lib's SO(3) `Kappa`. **Labels are doubled: `SS=2S`,
`TT=2T`.** Call `u4.Zeta(P1,P2,P3,SS,TT)` or the irrep overload
`u4.Zeta(irrep,SS,TT)`.

`Zeta` is guarded: it returns 0 for any `(SS,TT)` off the admissible,
parity-correct grid of the irrep, so it is safe to probe — but do not treat a 0
as an error. To enumerate the states that actually occur, prefer
[phys-list](../phys-list/SKILL.md) (`physicalBasis`) over scanning `Zeta`.

```python
import u4lib as u4
u4.initialize()
try:
    irr = u4.irrep(1, 0, 1)                 # adjoint
    assert u4.Zeta(irr, 2, 2) == 1          # (S,T)=(1,1) occurs once
    # (2,0,0): (S,T)=(1,1) and (0,0) only; (S,T)=(1,0) is absent
    assert u4.Zeta(u4.irrep(2,0,0), 2, 0) == 0
    # dimension sum rule over the valid grid
    grid = [(ss,tt) for ss in range(0,5) for tt in range(0,5)]
    dim = sum(u4.Zeta(irr,ss,tt)*(ss+1)*(tt+1) for ss,tt in grid)
    assert dim == irr.dim() == 15
finally:
    u4.finalize()
```

For a branching table, omit zero multiplicities unless requested, and label
whether you report `S,T` or doubled `SS,TT`. Each `zeta` copy contains
`(2S+1)(2T+1)` magnetic states; the returned count is not a zeta index.
See [basis definitions](../../../include/U4basis.h).
