---
name: phys-to-cano
description: Expand a physical SU(2)xSU(2) spin-isospin state |(P1,P2,P3) zeta S MS T MT> in the canonical U(4) Gelfand basis with u4lib, honoring the phase conventions.
---

# Physical-to-canonical transformation

Read [shared runtime instructions](../runtime.md). Use the free functions in
[U4SU2SU2_to_U4U3U2U1.h](../../../include/U4SU2SU2_to_U4U3U2U1.h):

```cpp
std::pair<std::vector<int>, Eigen::MatrixXd>
u4::phys_to_cano(const u4::physicalvector& p, int MMS, int MMT,
                 const std::vector<u4::gelfand>& gArr);

std::pair<std::vector<std::vector<int>>, std::vector<Eigen::MatrixXd>>
u4::phys_to_cano_allM(const u4::physicalvector& p,
                      const std::vector<u4::gelfand>& gArr);
```

They express the physical spin-isospin state `|(P1,P2,P3) zeta S MS T MT>` in the
canonical U(4)>U(3)>U(2)>U(1) Gelfand basis. **All angular labels are doubled:
`MMS=2MS`, `MMT=2MT`** (and the state's `SS=2S`, `TT=2T`). This is the basis
change `phys-coupling` and `phys-generators` call internally; it is **not** SU(4)
conjugation ([conjugation](../conjugation/SKILL.md)) and not a generator action
([cano-generators](../cano-generators/SKILL.md)). The highest-weight step solves
a null space, so call `u4::initialize(&argc,&argv)` first and `u4::finalize()`
after. The driver
[src/U4SU2SU2_to_U4U3U2U1.cpp](../../../src/U4SU2SU2_to_U4U3U2U1.cpp) is the
reference usage.

## Inputs, output and conventions

- Build `gArr = u4::gelfandBasis(p.irr())` and
  `p = u4::physicalvector(irrep, SS, TT)`; require `p.zeta()>0` and
  `|MMS|<=SS`, `|MMT|<=TT`.
- `phys_to_cano(p,MMS,MMT,gArr)` returns `(canoBasis, T)`: `canoBasis` are
  indices into `gArr` of the Gelfand states at this `(MS,MT)`, and `T` is
  `(canoBasis.size() x p.zeta())`. **Column `k` is the expansion of the k-th
  spin-isospin multiplicity copy** `zeta=k+1`; row `r` is the amplitude on
  `gArr[canoBasis[r]]`. `phys_to_cano_allM` returns the same for every
  `(MS,MT)`.
- Phase convention (documented in the header; do not alter): the highest weight
  `|... S S T T>` is the null space of the `S+ = E13+E24` and `T+ = E12+E34`
  projections; `S-` and `T-` fix the lower-M phases within a zeta multiplet.
  For `P1 != P3` the construction preserves zeta with
  `C|R zeta S MS T MT> = eps_R (-1)^(S-MS+T-MT) |barR zeta S,-MS T,-MT>`
  (`eps_R = (-1)^(P1+P2+P3)` for `P1>P3`, `(-1)^P2` for `P1<P3`). For `P1 = P3`
  conjugation acts as a projection-independent symmetric matrix that may **mix**
  zeta (e.g. `R=(1,1,1)`, `S=T=1`); do not realign the zeta basis to another
  convention — any zeta rotation must also be applied to coupling coefficients.

## Executable driver and checks

Drop this in `src/` (or compile against the library's configured flags). The
orthonormality of the columns is a convention-independent check.

```cpp
#include <cassert>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <u4lib.h>
#include <U4basis.h>
#include <U4SU2SU2_to_U4U3U2U1.h>

static bool close(double a, double b) { return std::abs(a - b) < 1e-5; }

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  u4::irrep ir(2, 0, 0);
  auto gArr = u4::gelfandBasis(ir);
  u4::physicalvector p(ir, 2, 2);          // |(2,0,0) S=1 T=1>, zeta_max = 1

  // Highest weight MMS=SS=2, MMT=TT=2: a single Gelfand state, amplitude 1.
  auto [cbHW, THW] = u4::phys_to_cano(p, 2, 2, gArr);
  assert(THW.rows() == 1 && THW.cols() == 1);
  assert(close(THW(0, 0), 1.0));

  // Columns are orthonormal for every (MS,MT): T^T T = I_zeta.
  auto [cb0, T0] = u4::phys_to_cano(p, 0, 0, gArr);
  Eigen::MatrixXd gram = T0.transpose() * T0;
  assert(close(gram(0, 0), 1.0));

  u4::finalize();
}
```

## Presentation and checks

Return, per requested `(MS,MT)`, the Gelfand states (`canoBasis` into `gArr`)
with the amplitude matrix, labelling rows by state and columns by `zeta`
(1-based). State that this is the reduced `(MS,MT)`-expansion, not a coupling
coefficient. Verify `T^T T = I_zeta` (orthonormal multiplicity columns) at each
`(MS,MT)`, that the highest-weight block is the single highest-weight state, and
that `phys_to_cano_allM` reproduces the per-projection results. Do not rescale
columns or drop the zeta mixing of a self-conjugate irrep to force a diagonal
form.
