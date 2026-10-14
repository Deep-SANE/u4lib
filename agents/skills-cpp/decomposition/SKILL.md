---
name: decomposition
description: Enumerate the full SU(4)/U(4) Kronecker product series A x B, listing every output irrep with its outer multiplicity, using u4lib (C++).
---

# Tensor-product (Clebsch-Gordan) decomposition (C++)

Read [shared runtime instructions](../runtime.md). Use the free function in
[U4decomposition.h](../../../include/U4decomposition.h):

```cpp
std::pair<std::vector<u4::irrep>, std::vector<int>>
u4::decomposition(const u4::irrep& A, const u4::irrep& B);
```

It returns parallel vectors: the output irreps `C` in `A x B` and their outer
multiplicities `etamax`. This is the **whole series**; for the multiplicity of a
single fixed `C` use [outer-mult](../outer-mult/SKILL.md) (`u4::Eta(A,B,C)`),
which equals the matching `etamax` entry. The algebra needs no coefficient
database, but calling it inside an initialized session is fine.

## Labels, ordering and conventions

- `u4::decomposition` takes and returns `u4::irrep` objects, printed as U(4) labels
  `[n14,n24,n34,n44]`; it reorders its two inputs by dimension internally, which
  does not change the set. Convert to SU(4) `(P1,P2,P3)` for reporting when the
  user works in SU(4) labels.
- The returned order is the generation order, **not** sorted; present every
  `(C, etamax)` pair with explicit labels.
- `etamax` is a count (outer multiplicity); valid eta indices are `0..etamax-1`
  when you later request coupling coefficients. A pair appears once with its
  full multiplicity, not repeated.

## Executable driver and checks

Drop this in `src/` (or compile against the library's configured flags, per the
runtime) and run it. The dimension sum rule is the decisive check.

```cpp
#include <cassert>
#include <vector>
#include <u4lib.h>
#include <U4decomposition.h>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  // Two adjoints (1,0,1) x (1,0,1) [15 x 15 = 225]:
  // Demonstrates non-trivial outer multiplicity (eta = 2 for C = (1,0,1)).
  u4::irrep A(1, 0, 1), B(1, 0, 1);
  auto [Cs, etas] = u4::decomposition(A, B);

  long total = 0;
  for (std::size_t i = 0; i < Cs.size(); ++i)
  {
    total += static_cast<long>(etas[i]) * Cs[i].dim();
    assert(etas[i] == u4::Eta(A, B, Cs[i]));     // matches the outer-mult skill
  }
  assert(Cs.size() == 6);
  assert(total == A.dim() * B.dim());            // 84 + 45 + 45 + 20 + 2*15 + 1 = 225
  assert(u4::Eta(A, B, u4::irrep(1, 0, 1)) == 2);

  u4::finalize();
}
```

## Presentation and checks

Return the list of `(C, etamax)` with labels, and state whether labels are SU(4)
`(P1,P2,P3)` or U(4) labels. Always report
`sum_C etamax*dim(C) == dim(A)*dim(B)`; a mismatch means the enumeration is
incomplete or an input was invalid. The set is symmetric under `A<->B`, and the
conjugate series is the irrep-conjugate set. Do not collapse an `etamax>1` entry
to one copy or silently restrict the output labels.
