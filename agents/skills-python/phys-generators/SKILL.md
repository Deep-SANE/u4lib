---
name: phys-generators
description: Compute SU(2)xSU(2)-reduced matrix elements of the physical SU(4) generators S, T and U between physical spin-isospin states with u4lib.
---

# Physical generator RMEs (S, T and U)

Read [shared runtime instructions](../runtime.md). `u4.RME_S(bra, ket)`,
`u4.RME_T(bra, ket)` and `u4.RME_U(bra, ket)` return
`<(P1,P2,P3) zeta' S' T' || G || (P1,P2,P3) zeta S T>` for the spin `S`
(rank (1,0)), isospin `T` (rank (0,1)), and `U` — the Gamow-Teller-type tensor
of rank (1,1) — with bra and ket in the **same** irrep. This is the SU(4)
analogue of u3lib's `L`/`Q`. **Labels are doubled: `SS=2S`, `TT=2T`.** Each
returns a tuple `(RME, bra_zeta, ket_zeta)`: `RME` is a `(bra.zeta() x ket.zeta())`
NumPy matrix indexed by zero-based multiplicity, and `bra_zeta`/`ket_zeta` are the
zero-based zeta labels `0..zeta-1` for the rows/columns. `RME_U` expands into the canonical
basis, so run inside a `u4.initialize()` / `u4.finalize()` session.

## Selection rules and conventions

- Build states with `u4.physicalBasis(irrep)` or `u4.physicalvector(irrep, SS, TT)`;
  require `zeta()>0`.
- `S` and `T` are diagonal in `(S,T)`: `RME_S` is zero unless bra and ket share
  `(SS,TT)`, and otherwise every entry of the `zeta' x zeta` block equals
  `sqrt(S(S+1)(2S+1)(2T+1))`; `RME_T` gives `sqrt(T(T+1)(2T+1)(2S+1))`. Report the
  block as returned — do not diagonalize.
- `U` is rank (1,1): with `ss = bra.SS()-ket.SS()`, `tt = bra.TT()-ket.TT()`, it
  is zero unless `ss, tt in {-2,0,+2}` and the spin/isospin triangles hold. A
  `ket` with `S=0` or `T=0` divides by zero in the components that need it —
  report such a channel as forbidden rather than indexing it.
- The reduced-matrix-element convention carries an extra `sqrt((2S'+1)(2T'+1))`
  (bra); state this when comparing with another source. Keep the library's signs.

```python
import math
import u4lib as u4
u4.initialize()
try:
    basis = u4.physicalBasis(u4.irrep(2, 0, 0))   # (S,T)=(1,1) [0], (0,0) [1]
    st11, st00 = basis[0], basis[1]
    # S and T on (S=1,T=1): sqrt(S(S+1)(2S+1)(2T+1)) = sqrt(18)
    assert abs(u4.RME_S(st11, st11)[0][0, 0] - math.sqrt(18)) < 1e-4
    assert abs(u4.RME_T(st11, st11)[0][0, 0] - math.sqrt(18)) < 1e-4
    # U diagonal (ss=tt=0) component on (S=1,T=1) is 3
    assert abs(u4.RME_U(st11, st11)[0][0, 0] - 3.0) < 1e-4
    # S is diagonal in (S,T): between (1,1) and (0,0) it vanishes
    assert abs(u4.RME_S(st11, st00)[0][0, 0]) < 1e-9
finally:
    u4.finalize()
```

Report the matrix with its row (`zeta'`) and column (`zeta`) labels; a full
element additionally needs the spin and isospin CG factors for the requested
projections. A zero inside an allowed block is a valid element.
