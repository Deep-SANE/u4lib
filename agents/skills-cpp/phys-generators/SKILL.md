---
name: phys-generators
description: Compute SU(2)xSU(2)-reduced matrix elements of the physical SU(4) generators S, T and U between physical spin-isospin states with u4lib.
---

# Physical generator RMEs (S, T and U)

Read [shared runtime instructions](../runtime.md). Use the free functions in
[U4generatorsRME.h](../../../include/U4generatorsRME.h):

```cpp
Eigen::MatrixXd u4::RME_S(u4::physicalvector& bra, u4::physicalvector& ket,
                          std::vector<int>& bra_zeta, std::vector<int>& ket_zeta);
Eigen::MatrixXd u4::RME_T(/* same signature */);
Eigen::MatrixXd u4::RME_U(/* same signature */);
```

They compute `<(P1,P2,P3) zeta' S' T' || G || (P1,P2,P3) zeta S T>` for the three
physical SU(4) generators with bra and ket in the **same** irrep: the spin `S`
(rank (1,0) in (spin,isospin)), the isospin `T` (rank (0,1)), and `U` — the
15th-generator Gamow-Teller-type tensor of rank (1,1). This is the SU(4)
analogue of u3lib's `L`/`Q`. **Labels are doubled: `SS=2S`, `TT=2T`.** Each
returns an `Eigen::MatrixXd` of shape `(bra.zeta() x ket.zeta())` and fills the
out-parameters `bra_zeta`/`ket_zeta` with the 1-based zeta labels. `RME_U`
expands into the canonical basis, so call `u4::initialize(&argc,&argv)` first and
`u4::finalize()` after; the driver
[src/U4generatorsRME.cpp](../../../src/U4generatorsRME.cpp) is the reference
usage.

## Inputs, shape and convention

- Build states with `u4::physicalBasis(irrep)` or `u4::physicalvector(irrep, SS, TT)`;
  require `zeta()>0`. The result is an `Eigen::MatrixXd` of shape
  `(bra.zeta(), ket.zeta())`, indexed `RME(i,j)` by **zero-based** zeta position.
  The out-parameters `bra_zeta`/`ket_zeta` are **filled** with the **1-based**
  zeta labels for presentation; do not preset them.
- The reduced-matrix-element convention carries an extra `sqrt((2S'+1)(2T'+1))`
  factor (bra), stated in the header. State this when comparing with another
  source.

### S and T

`S` is diagonal in `(S,T)`: `RME_S` is zero unless `bra` and `ket` share
`(SS,TT)`, and otherwise every entry of the `zeta' x zeta` block equals
`sqrt(S(S+1)(2S+1)(2T+1))`. Likewise `RME_T` gives `sqrt(T(T+1)(2T+1)(2S+1))`.
The library fills the whole zeta block with that one value; report it as
returned — do not diagonalize.

### U

`U` is a rank-(1,1) tensor. Define `ss = bra.SS()-ket.SS()`, `tt = bra.TT()-ket.TT()`;
`RME_U` is zero unless `ss, tt in {-2,0,+2}` (i.e. ΔS, ΔT in {-1,0,+1}) and the
spin/isospin triangles hold. The library forms the `(+2,+2)`, `(+2,0)`, `(0,0)`
and `(-2,+2)` components directly from the canonical expansion, and the
remaining components by the transpose/phase relations in the header. Domain
guard: several component factors divide by `ket.SS()` or `ket.TT()`, so a
`ket` with `S=0` or `T=0` in those components divides by zero — report such a
channel as forbidden rather than indexing it.

## Executable driver and checks

Drop this in `src/` (or compile against the library's configured flags) and run.

```cpp
#include <cassert>
#include <cmath>
#include <vector>
#include <u4lib.h>
#include <U4basis.h>
#include <U4generatorsRME.h>

static bool close(double a, double b) { return std::abs(a - b) < 1e-4; }

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  // (2,0,0) has (S,T) = (1,1) [index 0] and (0,0) [index 1].
  auto basis = u4::physicalBasis(u4::irrep(2, 0, 0));
  u4::physicalvector st11 = basis[0];      // (SS,TT) = (2,2)
  std::vector<int> a, b;

  // S and T on (S=1,T=1): sqrt(S(S+1)(2S+1)(2T+1)) = sqrt(1*2*3*3) = sqrt(18).
  assert(close(u4::RME_S(st11, st11, a, b)(0, 0), std::sqrt(18.0)));  a.clear(); b.clear();
  assert(close(u4::RME_T(st11, st11, a, b)(0, 0), std::sqrt(18.0)));  a.clear(); b.clear();

  // U, diagonal (ss=tt=0) component on (S=1,T=1) is 3.
  assert(close(u4::RME_U(st11, st11, a, b)(0, 0), 3.0));             a.clear(); b.clear();

  // S is diagonal in (S,T): between (1,1) and (0,0) it vanishes.
  u4::physicalvector st00 = basis[1];      // (SS,TT) = (0,0)
  assert(close(u4::RME_S(st11, st00, a, b)(0, 0), 0.0));

  u4::finalize();
}
```

## Presentation and checks

Report the matrix with its row (`zeta'`) and column (`zeta`) labels and the
bra/ket `(P1,P2,P3) zeta_max S T`. State whether you show reduced or full matrix
elements: a full element additionally needs the spin and isospin CG factors for
the requested projections. For `S`/`T` verify the single value and the
off-`(S,T)` zero; for `U` verify the tensor selection, the component
transpose/phase relations, and that a forbidden channel is reported rather than
returned as a divide-by-zero. A zero inside an allowed block is a valid element.
