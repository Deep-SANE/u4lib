---
name: phys-to-cano
description: Expand a physical SU(2)xSU(2) spin-isospin state |(P1,P2,P3) zeta S MS T MT> in the canonical U(4) Gelfand basis with u4lib, honoring the phase conventions.
---

# Physical-to-canonical transformation

Read [shared runtime instructions](../runtime.md). `u4.phys_to_cano(p, MMS, MMT, gArr)`
expresses the physical spin-isospin state `|(P1,P2,P3) zeta S MS T MT>` in the
canonical U(4)>U(3)>U(2)>U(1) Gelfand basis; `u4.phys_to_cano_allM(p, gArr)` does
every projection at once. **All angular labels are doubled: `MMS=2MS`, `MMT=2MT`**
(and the state's `SS=2S`, `TT=2T`). This is the basis change
[phys-coupling](../phys-coupling/SKILL.md) and
[phys-generators](../phys-generators/SKILL.md) use internally — not SU(4)
conjugation ([conjugation](../conjugation/SKILL.md)) and not a generator action
([cano-generators](../cano-generators/SKILL.md)). The highest-weight step solves
a null space, so run inside a `u4.initialize()` / `u4.finalize()` session.

## Inputs, output and conventions

- Build `gArr = u4.gelfandBasis(p.irr())` and `p = u4.physicalvector(irrep, SS, TT)`;
  require `p.zeta()>0` and `|MMS|<=SS`, `|MMT|<=TT`.
- `phys_to_cano(p, MMS, MMT, gArr)` returns `(canoBasis, T)`: `canoBasis` are
  indices into `gArr` of the Gelfand states at this `(MS,MT)`, and **column `k`
  of `T` is the k-th spin-isospin multiplicity copy** (`zeta=k+1`), row `r` the
  amplitude on `gArr[canoBasis[r]]`. `phys_to_cano_allM` returns the same for
  every `(MS,MT)`.
- Phase convention (documented in the header; do not alter): the highest weight
  is the null space of the `S+ = E13+E24` and `T+ = E12+E34` projections; `S-`
  and `T-` fix the lower-M phases within a zeta multiplet. For a self-conjugate
  irrep (`P1 = P3`, e.g. `R=(1,1,1)`, `S=T=1`) conjugation acts as a symmetric
  matrix that may **mix** zeta; do not realign the zeta basis. Any zeta rotation
  must also be applied to coupling coefficients.

```python
import u4lib as u4
u4.initialize()
try:
    ir = u4.irrep(2, 0, 0)
    gArr = u4.gelfandBasis(ir)
    p = u4.physicalvector(ir, 2, 2)               # |(2,0,0) S=1 T=1>, zeta_max = 1
    # Highest weight MMS=SS=2, MMT=TT=2: a single Gelfand state, amplitude 1.
    cbHW, THW = u4.phys_to_cano(p, 2, 2, gArr)
    assert THW.shape == (1, 1) and abs(THW[0, 0] - 1.0) < 1e-5
    # Columns orthonormal for every (MS,MT): T^T T = I_zeta.
    cb0, T0 = u4.phys_to_cano(p, 0, 0, gArr)
    assert abs((T0.T @ T0)[0, 0] - 1.0) < 1e-5
finally:
    u4.finalize()
```

Verify `T^T T = I_zeta` at each `(MS,MT)`; state this is the expansion at fixed
`(MS,MT)`, not a coupling coefficient. Do not rescale columns or drop the zeta
mixing of a self-conjugate irrep.
