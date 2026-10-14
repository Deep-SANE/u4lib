---
name: u6recoupling
description: Compute U-type three-irrep SU(4) recoupling blocks or complete multiplicity-resolved matrices (C++).
---

# u6recoupling (C++)

Read [shared runtime instructions](../runtime.md) for the execution setup and build flags.

Routes: `(A B)AB C -> T` and `A (B C)BC -> T`. Compute the four `u4::Eta` counts (all positive), then call `get_U4U6` — note **T precedes C** in the argument order:
`get_U4U6(A, B, T, C, AB, eAB, eAB_C, BC, eBC, eA_BC)`.
Accessor: `block(iAB, iAB_C, iBC, iA_BC)`, each index bounded by its count.

```cpp
#include <u4lib.h>
#include <cassert>
#include <cmath>
#include <vector>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  // Example with non-trivial outer multiplicities (eta = 2 on all 4 channels):
  // Three adjoints (1,0,1) coupling to (1,0,1) through intermediate (1,0,1):
  u4::irrep A(1, 0, 1), B(1, 0, 1), C(1, 0, 1), T(1, 0, 1);
  u4::irrep AB(1, 0, 1), BC(1, 0, 1);

  int eAB   = u4::Eta(A, B, AB);    // (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
  int eAB_C = u4::Eta(AB, C, T);    // (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
  int eBC   = u4::Eta(B, C, BC);    // (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
  int eA_BC = u4::Eta(A, BC, T);    // (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
  assert(eAB == 2 && eAB_C == 2 && eBC == 2 && eA_BC == 2);

  // Getter call (note T precedes C):
  auto block = u4::u4u6db().get_U4U6(A, B, T, C, AB, eAB, eAB_C, BC, eBC, eA_BC);
  assert(block.size() == 16);
  assert(std::abs(block(0, 0, 0, 0) - 0.6875) < 1e-10);            // 11/16
  assert(std::abs(block(0, 0, 0, 1) - 0.3247595264) < 1e-8);       // 3*sqrt(3)/16
  assert(std::abs(block(0, 0, 1, 1) - 0.0625) < 1e-10);            // 1/16
  assert(std::abs(block(1, 1, 1, 1) - (41.0 / 48.0)) < 1e-10);     // 41/48

  // Dynamic channel-discovery pattern for complete matrix assembly:
  struct Channel { u4::irrep inter; int e1, e2; };
  std::vector<Channel> left_channels;
  auto [ab_irreps, ab_mults] = u4::decomposition(A, B);
  for (size_t i = 0; i < ab_irreps.size(); ++i) {
    int n2 = u4::Eta(ab_irreps[i], C, T);
    for (int e1 = 0; e1 < ab_mults[i]; ++e1)
      for (int e2 = 0; e2 < n2; ++e2)
        left_channels.push_back({ab_irreps[i], e1, e2});
  }
  assert(!left_channels.empty());

  u4::finalize();
}
```

For complete change-of-coupling matrix assembly:
1. Enumerate left route channels `(AB, eta_AB, eta_AB_C)` using `u4::decomposition(A, B)` and `u4::Eta(AB, C, T)`.
2. Enumerate right route channels `(BC, eta_BC, eta_A_BC)` using `u4::decomposition(B, C)` and `u4::Eta(A, BC, T)`.
3. Order channels lexicographically and check row/column orthogonality of the assembled matrix.
See [test/recoupling_cases.h](../../../test/recoupling_cases.h) for full matrix checks.

### Parallel batch (OpenMP)

```cpp
#include <omp.h>
#include <u4lib.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  struct U6Task {
    u4::irrep A, B, T, C, AB, BC;
    int eAB, eAB_C, eBC, eA_BC;
  };
  u4::irrep adj(1, 0, 1), sc(0, 0, 0);
  std::vector<U6Task> batch = {
    {sc, sc, sc, sc, sc, sc, 1, 1, 1, 1},
    {adj, adj, adj, adj, adj, adj, 2, 2, 2, 2}
  };
  std::vector<double> results(batch.size());

#pragma omp parallel
  {
    u4::initialize_thread();
#pragma omp for
    for (int i = 0; i < static_cast<int>(batch.size()); ++i) {
      const auto& t = batch[i];
      // Note argument order: T precedes C
      auto blk = u4::u4u6db().get_U4U6(t.A, t.B, t.T, t.C,
                                        t.AB, t.eAB, t.eAB_C,
                                        t.BC, t.eBC, t.eA_BC);
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
