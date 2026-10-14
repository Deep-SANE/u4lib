---
name: phys-coupling
description: Compute SU(2)xSU(2)-reduced physical coupling coefficients with zeta and eta multiplicities (C++).
---

# phys-coupling (C++)

Read [shared runtime instructions](../runtime.md) before execution. The C++
examples below assume the `u4::` API and the lifecycle described there.

These are the physical spin-isospin reduced coupling coefficients
`(A, SA TA) x (B, SB TB) -> (C, SC TC)`, the SU(4) analogue of u3lib's
SO(3)-reduced `WigU3SO3`. **Spin/isospin labels are doubled: `SS=2S`, `TT=2T`.**
For each leg validate the irrep and that `zeta = Zeta(irrep, SS, TT) > 0`
(use [phys-list](../phys-list/SKILL.md) to enumerate valid `(S,T)`), require
`Eta(A,B,C) > 0`, and the SU(2) triangle conditions on both spin and isospin.

```cpp
#include <u4lib.h>
#include <cassert>
#include <cmath>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  // Coupling (1,0,1) x (1,1,1) -> (1,1,1) with (S,T) = (1,1) [SS=2, TT=2]:
  // Demonstrates inner spin-isospin multiplicity (zeta = 2) and outer multiplicity (eta = 3).
  u4::irrep A(1, 0, 1), B(1, 1, 1), C(1, 1, 1);
  int eta = u4::Eta(A, B, C);                     // eta = 3
  assert(eta == 3);
  assert(u4::Zeta(A, 2, 2) == 1);                 // zetaA = 1
  assert(u4::Zeta(B, 2, 2) == 2);                 // zetaB = 2
  assert(u4::Zeta(C, 2, 2) == 2);                 // zetaC = 2

  auto blk = u4::wigU4SU2SU2db().get_U4SU2SU2(1, 0, 1, 2, 2,
                                              1, 1, 1, 2, 2,
                                              1, 1, 1, 2, 2);

  // Shape: (zetaA, zetaB, zetaC, eta) = (1, 2, 2, 3), total 12 entries
  assert(blk.zetaAmax() == 1 && blk.zetaBmax() == 2 && blk.zetaCmax() == 2 && blk.etamax() == 3);

  // Verified coefficients: blk(zetaA, zetaB, zetaC, eta)
  assert(std::abs(blk(0, 0, 0, 0) - 0.2434446675) < 1e-8);
  assert(std::abs(blk(0, 0, 1, 0) - (-0.2655760009)) < 1e-8);
  assert(std::abs(blk(0, 1, 1, 0) - (-0.4647580015)) < 1e-8);
  assert(std::abs(blk(0, 0, 0, 2) - 0.4123538702) < 1e-8);

  u4::finalize();
}
```

Prefer the structured `get_U4SU2SU2` accessor over guessing how the flat array
reshapes; axes are `(zetaA, zetaB, zetaC, eta)`. Report reduced coefficients;
full magnetic CGs additionally require the SU(2) spin **and** isospin CG factors
for the requested `(MS, MT)`. For orthogonality, sum over the complete required
channel set, not one block. See [physical checks](../../../test/physical_cases.h).

### Parallel batch (OpenMP)

```cpp
#include <omp.h>
#include <u4lib.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  struct PhysTask {
    int p1a, p2a, p3a, ss1, tt1;
    int p1b, p2b, p3b, ss2, tt2;
    int p1c, p2c, p3c, ss3, tt3;
  };
  std::vector<PhysTask> batch = {
    {1, 0, 0, 1, 1,  0, 0, 1, 1, 1,  1, 0, 1, 2, 2},
    {1, 0, 1, 2, 2,  1, 1, 1, 2, 2,  1, 1, 1, 2, 2}
  };
  std::vector<double> results(batch.size());

#pragma omp parallel
  {
    u4::initialize_thread();
#pragma omp for
    for (int i = 0; i < static_cast<int>(batch.size()); ++i) {
      const auto& t = batch[i];
      auto blk = u4::wigU4SU2SU2db().get_U4SU2SU2(t.p1a, t.p2a, t.p3a, t.ss1, t.tt1,
                                                  t.p1b, t.p2b, t.p3b, t.ss2, t.tt2,
                                                  t.p1c, t.p2c, t.p3c, t.ss3, t.tt3);
#pragma omp critical
      results[i] = blk(0, 0, 0, 0);
    }
    u4::finalize_thread();
  }

  for (size_t i = 0; i < results.size(); ++i)
    std::cout << "[" << i << "] = " << results[i] << "\n";

  u4::finalize();
}
```
