---
name: inner-mult
description: Calculate the SU(2)xSU(2) spin-isospin branching multiplicity zeta of (S,T) in an SU(4) irrep with u4lib.
---

# inner-mult

Read [shared runtime instructions](../runtime.md) before execution. The C++
examples below assume the `u4::` API and the lifecycle described there.

Here inner multiplicity means SU(4) > SU(2)xSU(2) (spin x isospin) branching:
the multiplicity `zeta` of a spin-isospin state `(S,T)` in the irrep. It is the
SU(4) analogue of u3lib's SO(3) `Kappa`. **Labels are doubled: `SS=2S`,
`TT=2T`.** Call `u4::Zeta(P1,P2,P3,SS,TT)` or the irrep overload
`u4::Zeta(irrep,SS,TT)`.

`Zeta` is guarded: it returns 0 for any `(SS,TT)` off the admissible,
parity-correct grid of the irrep, so it is safe to probe — but do not treat a 0
as an error. To enumerate the states that actually occur, prefer
[phys-list](../phys-list/SKILL.md) (`physicalBasis`) over scanning `Zeta`.

```cpp
u4::irrep irr(1, 0, 1);                       // adjoint
assert(u4::Zeta(irr, 2, 2) == 1);             // (S,T)=(1,1) occurs once
assert(u4::Zeta(u4::irrep(2, 0, 0), 2, 0) == 0);   // (S,T)=(1,0) absent
// (1,1,1) exhibits non-trivial inner multiplicity zeta = 2 at (S,T)=(1,1):
assert(u4::Zeta(u4::irrep(1, 1, 1), 2, 2) == 2);
long dim = 0;
for (int ss = 0; ss <= 4; ++ss)
  for (int tt = 0; tt <= 4; ++tt)
    dim += (long)u4::Zeta(irr, ss, tt) * (ss + 1) * (tt + 1);
assert(dim == irr.dim() && irr.dim() == 15);
```

For a branching table, omit zero multiplicities unless requested, and label
whether you report `S,T` or doubled `SS,TT`. Each `zeta` copy contains
`(2S+1)(2T+1)` magnetic states; the returned count is not a zeta index.
See [basis definitions](../../../include/U4basis.h).
