---
name: z6recoupling
description: Compute Z-type three-irrep SU(4) recoupling between AB-C and AC-B coupling routes (C++).
---

# z6recoupling (C++)

Read [shared runtime instructions](../runtime.md) for the execution setup and build flags.

Routes: `(A B)AB C -> T` and `(A C)AC B -> T`. Compute the four `u4::Eta` counts (all positive), then call `get_U4Z6` — the getter **starts with B, A** (exchanging the first pair):
`get_U4Z6(B, A, T, C, AB, eAB, eAB_C, AC, eAC, eAC_B)`.
Accessor: `block(iAB, iAB_C, iAC, iAC_B)`. The container shape getters `rhoACmax()`/`rhoAC_Bmax()` count the AC and AC_B channels, respectively.

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
  u4::irrep AB(1, 0, 1), AC(1, 0, 1);

  int eAB   = u4::Eta(A, B, AB);    // (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
  int eAB_C = u4::Eta(AB, C, T);    // (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
  int eAC   = u4::Eta(A, C, AC);    // (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
  int eAC_B = u4::Eta(AC, B, T);    // (1,0,1) x (1,0,1) -> (1,0,1) has eta = 2
  assert(eAB == 2 && eAB_C == 2 && eAC == 2 && eAC_B == 2);

  // Getter call (note leading B, A, and T precedes C):
  auto block = u4::u4z6db().get_U4Z6(B, A, T, C, AB, eAB, eAB_C, AC, eAC, eAC_B);
  assert(block.size() == 16);
  assert(std::abs(block(0, 0, 0, 0) - (-0.0625)) < 1e-10);         // -1/16
  assert(std::abs(block(0, 1, 1, 0) - 0.8125) < 1e-10);            // 13/16
  assert(std::abs(block(0, 0, 1, 1) - (-0.1875)) < 1e-10);         // -3/16
  assert(std::abs(block(1, 1, 1, 1) - (5.0 / 48.0)) < 1e-10);      // 5/48

  // Dynamic channel-discovery pattern for complete matrix assembly:
  struct Channel { u4::irrep inter; int e1, e2; };
  std::vector<Channel> left_channels, right_channels;
  auto [ab_irreps, ab_mults] = u4::decomposition(A, B);
  for (size_t i = 0; i < ab_irreps.size(); ++i) {
    int n2 = u4::Eta(ab_irreps[i], C, T);
    for (int e1 = 0; e1 < ab_mults[i]; ++e1)
      for (int e2 = 0; e2 < n2; ++e2)
        left_channels.push_back({ab_irreps[i], e1, e2});
  }
  auto [ac_irreps, ac_mults] = u4::decomposition(A, C);
  for (size_t i = 0; i < ac_irreps.size(); ++i) {
    int n2 = u4::Eta(ac_irreps[i], B, T);
    for (int e1 = 0; e1 < ac_mults[i]; ++e1)
      for (int e2 = 0; e2 < n2; ++e2)
        right_channels.push_back({ac_irreps[i], e1, e2});
  }
  assert(!left_channels.empty() && !right_channels.empty());

  u4::finalize();
}
```

For complete change-of-coupling matrix assembly:
1. Enumerate left route channels `(AB, eta_AB, eta_AB_C)` using `u4::decomposition(A, B)` and `u4::Eta(AB, C, T)`.
2. Enumerate right route channels `(AC, eta_AC, eta_AC_B)` using `u4::decomposition(A, C)` and `u4::Eta(AC, B, T)`.
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

  struct Z6Task {
    u4::irrep A, B, T, C, AB, AC;
    int eAB, eAB_C, eAC, eAC_B;
  };
  u4::irrep adj(1, 0, 1), sc(0, 0, 0);
  std::vector<Z6Task> batch = {
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
      // Note argument order: B precedes A, T precedes C
      auto blk = u4::u4z6db().get_U4Z6(t.B, t.A, t.T, t.C,
                                        t.AB, t.eAB, t.eAB_C,
                                        t.AC, t.eAC, t.eAC_B);
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
