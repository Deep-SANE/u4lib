---
name: u9recoupling
description: Compute four-irrep SU(4) 9-coefficient recoupling with all six multiplicity axes and correct result lifetime (C++).
---

# u9recoupling (C++)

Read [shared runtime instructions](../runtime.md) for the execution setup and build flags.

Routes: `(A B)AB (C D)CD -> T` and `(A C)AC (B D)BD -> T`. Compute all six multiplicities with `u4::Eta` (all must be positive), then call `u4::u4u9db().get_U4U9(...)`.
Accessor: `block(iAC, iBD, iLeft, iAB, iCD, iRight)`.
* **Important Axis Ordering:** The 3rd axis is the left route `(AB)(CD) -> T`, and the 6th is the right route `(AC)(BD) -> T`. Keep them distinct, especially when their sizes differ; an internal assertion catches any swap.

**Result lifetime.** With the U9 cache enabled (the default), `get_U4U9` returns a span-backed view into a scratch buffer that the next U9 database call overwrites; the cache-disabled build returns owning storage. To preserve a block across subsequent database calls (batching), copy its values into an owning `u4::U4U9<u4::coeffs_t>`.

```cpp
#include <u4lib.h>
#include <cassert>
#include <cmath>
#include <utility>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  // Four adjoint irreps (1,0,1) coupling to total scalar (0,0,0) via (1,0,1):
  // Demonstrates non-trivial outer multiplicities (eta = 2 on four axes).
  u4::irrep A(1, 0, 1), B(1, 0, 1), C(1, 0, 1), D(1, 0, 1);
  u4::irrep AB(1, 0, 1), CD(1, 0, 1), AC(1, 0, 1), BD(1, 0, 1), T(0, 0, 0);

  int eAB = u4::Eta(A, B, AB);       // 2
  int eCD = u4::Eta(C, D, CD);       // 2
  int eAC = u4::Eta(A, C, AC);       // 2
  int eBD = u4::Eta(B, D, BD);       // 2
  int eLeft = u4::Eta(AB, CD, T);    // 1
  int eRight = u4::Eta(AC, BD, T);   // 1
  assert(eAB == 2 && eCD == 2 && eAC == 2 && eBD == 2);
  assert(eLeft == 1 && eRight == 1);

  // Getter arguments: (A, B, C, D, AB, CD, AC, BD, T, eAC, eBD, eLeft, eAB, eCD, eRight)
  const auto block = u4::u4u9db().get_U4U9(A, B, C, D, AB, CD, AC, BD, T,
                                           eAC, eBD, eLeft, eAB, eCD, eRight);
  assert(block.size() == 16);

  // Verified coefficients: block(iAC, iBD, iLeft, iAB, iCD, iRight)
  assert(std::abs(block(0, 0, 0, 0, 0, 0) - (-0.0625)) < 1e-10);      // -1/16
  assert(std::abs(block(0, 1, 0, 1, 0, 0) - 0.8125) < 1e-10);         // 13/16
  assert(std::abs(block(0, 1, 0, 0, 1, 0) - (-0.1875)) < 1e-10);      // -3/16
  assert(std::abs(block(1, 1, 0, 1, 1, 0) - (5.0 / 48.0)) < 1e-10);   // 5/48

  // Preserve coefficients in an owning container before any subsequent DB calls:
  u4::coeffs_t values(block.data().begin(), block.data().end());
  u4::U4U9<u4::coeffs_t> saved(std::move(values), eAC, eBD, eLeft, eAB, eCD, eRight);
  assert(std::abs(saved(1, 1, 0, 1, 1, 0) - (5.0 / 48.0)) < 1e-10);

  u4::finalize();
}
```

For a complete recoupling matrix:

1. Discover left-route pairs `(AB, CD)` using `u4::decomposition(A, B)` and
   `u4::decomposition(C, D)`; retain pairs with `u4::Eta(AB, CD, T) > 0`.
   Enumerate every `(iAB, iCD, iLeft)` for each retained pair.
2. Discover right-route pairs `(AC, BD)` using `u4::decomposition(A, C)` and
   `u4::decomposition(B, D)`; retain pairs with `u4::Eta(AC, BD, T) > 0`.
   Enumerate every `(iAC, iBD, iRight)` for each retained pair.
3. Label matrix rows by left-route channels and columns by right-route channels.
   Fill each entry from `block(iAC, iBD, iLeft, iAB, iCD, iRight)` for the
   corresponding four intermediate irreps. Preserve the library multiplicity basis.
4. Check row and column orthogonality only after assembling both complete channel
   sets. A block for fixed intermediate irreps is generally not orthogonal by itself.
See [test/recoupling_cases.h](../../../test/recoupling_cases.h) for full matrix checks.

### Parallel batch (OpenMP)

```cpp
#include <omp.h>
#include <u4lib.h>
#include <iostream>
#include <utility>
#include <vector>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  struct U9Task {
    u4::irrep A, B, C, D, AB, CD, AC, BD, T;
    int eAC, eBD, eLeft, eAB, eCD, eRight;
  };
  u4::irrep adj(1, 0, 1), sc(0, 0, 0);
  std::vector<U9Task> batch = {
    {sc, sc, sc, sc, sc, sc, sc, sc, sc, 1, 1, 1, 1, 1, 1},
    {adj, adj, adj, adj, adj, adj, adj, adj, sc, 2, 2, 1, 2, 2, 1}
  };
  std::vector<double> results(batch.size());

#pragma omp parallel
  {
    u4::initialize_thread();
#pragma omp for
    for (int i = 0; i < static_cast<int>(batch.size()); ++i) {
      const auto& t = batch[i];
      const auto view = u4::u4u9db().get_U4U9(t.A, t.B, t.C, t.D,
                                              t.AB, t.CD, t.AC, t.BD, t.T,
                                              t.eAC, t.eBD, t.eLeft,
                                              t.eAB, t.eCD, t.eRight);
      // Copy span-backed view into owning container before any subsequent DB calls
      u4::coeffs_t values(view.data().begin(), view.data().end());
      u4::U4U9<u4::coeffs_t> saved(std::move(values), t.eAC, t.eBD, t.eLeft,
                                   t.eAB, t.eCD, t.eRight);
#pragma omp critical
      results[i] = saved(0, 0, 0, 0, 0, 0);
    }
    u4::finalize_thread();
  }

  for (size_t i = 0; i < results.size(); ++i)
    std::cout << "[" << i << "] = " << results[i] << "\n";

  u4::finalize();
}
```
