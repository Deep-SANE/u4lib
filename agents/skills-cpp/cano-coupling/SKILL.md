---
name: cano-coupling
description: Compute SU(4)>U(3) reduced canonical coupling coefficients, including all rho and eta multiplicities (C++).
---

# cano-coupling (C++)

Read [shared runtime instructions](../runtime.md) before execution. The C++
examples below assume the `u4::` API and the lifecycle described there.

Each input state is seven labels: the U(4) row `[n14,n24,n34,n44]` and its U(3)
sub-row `[n13,n23,n33]`. Check top-row ordering `n14>=n24>=n34>=n44` and the
U(3) interlacing before any call.

**Quanta conservation:** In U(4), total quanta $N = \sum_{i=1}^4 n_{i4}$ is strictly
conserved under coupling: $N(A) + N(B) = N(C)$. For two normalized SU(4) $(1,1,1)$
irreps with $N=6$ ($[3,2,1,0]$), the coupled irrep $(1,2,1)$ must carry total quanta
$N=6+6=12$, corresponding to the determinant-shifted row $[5,4,2,1]$ (not the
normalized $[3,2,1,0]$). Similarly, U(3) sublabels must conserve quanta:
$\sum n_{i3}^A + \sum n_{i3}^B = \sum n_{i3}^C$.

`wigU4U3db().get(...)` returns SU(4)>U(3) **reduced** coefficients (not full
magnetic CGs) as a **2-D** `Eigen::MatrixXd` over `(rho, eta)`: `rho` is the U(3)
inner multiplicity and `eta = Eta(A,B,C)` the SU(4) outer multiplicity.

```cpp
#include <u4lib.h>
#include <cassert>
#include <cmath>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  // Coupling (1,1,1) [3,2,1,0] x (1,1,1) [3,2,1,0] -> (1,2,1) [5,4,2,1]:
  // Demonstrates non-trivial U(3) inner multiplicity (rho = 2) and SU(4) outer multiplicity (eta = 4).
  // Sublabels: subA = [3,2,1] (quanta 6), subB = [3,2,0] (quanta 5), subC = [5,4,2] (quanta 11).
  auto arr = u4::wigU4U3db().get(3, 2, 1, 0,  3, 2, 1,
                                 3, 2, 1, 0,  3, 2, 0,
                                 5, 4, 2, 1,  5, 4, 2);

  assert(arr.rows() == 2 && arr.cols() == 4);   // rho = 2, eta = 4
  assert(std::abs(arr(0, 0) - std::sqrt(2.0 / 3.0)) < 1e-8);   // sqrt(2/3) ~ 0.816497
  assert(std::abs(arr(0, 1)) < 1e-12);
  assert(std::abs(arr(0, 2) - 0.2381965337) < 1e-8);
  assert(std::abs(arr(1, 0)) < 1e-12);
  assert(std::abs(arr(1, 1) - std::sqrt(5.0 / 8.0)) < 1e-8);   // sqrt(5/8) ~ 0.790569

  u4::finalize();
}
```

Return every `(rho, eta)` entry or validate a requested index pair against the
array shape. These are reduced coefficients: the n11/magnetic projections are
absent; a full canonical CG additionally needs the subgroup U(3)>U(2)>U(1)
factor. The highest-weight null space, RREF, orthonormalization and
leading-sign choices fix the library `eta` basis — preserve it, and apply any
`eta`-basis change consistently to every coefficient carrying `eta`. Use scalar
identities or signed reference rows from
[canonical tests](../../../test/canonical_cases.h); an arbitrary reduced row need
not have unit norm by itself.

### Parallel batch (OpenMP)

```cpp
#include <omp.h>
#include <u4lib.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  struct CanoTask { int a[7], b[7], c[7]; };
  std::vector<CanoTask> batch = {
    {{1, 0, 0, 0,  1, 0, 0}, {0, 0, 0, 0,  0, 0, 0}, {1, 0, 0, 0,  1, 0, 0}},
    {{3, 2, 1, 0,  3, 2, 1}, {3, 2, 1, 0,  3, 2, 0}, {5, 4, 2, 1,  5, 4, 2}}
  };
  std::vector<double> results(batch.size());

#pragma omp parallel
  {
    u4::initialize_thread();
#pragma omp for
    for (int i = 0; i < static_cast<int>(batch.size()); ++i) {
      const auto& t = batch[i];
      auto arr = u4::wigU4U3db().get(t.a[0], t.a[1], t.a[2], t.a[3], t.a[4], t.a[5], t.a[6],
                                     t.b[0], t.b[1], t.b[2], t.b[3], t.b[4], t.b[5], t.b[6],
                                     t.c[0], t.c[1], t.c[2], t.c[3], t.c[4], t.c[5], t.c[6]);
#pragma omp critical
      results[i] = arr(0, 0);
    }
    u4::finalize_thread();
  }

  for (size_t i = 0; i < results.size(); ++i)
    std::cout << "[" << i << "] = " << results[i] << "\n";

  u4::finalize();
}
```
