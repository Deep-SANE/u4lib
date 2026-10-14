---
name: phys-list
description: Enumerate the physical SU(2)xSU(2) spin-isospin basis (zeta, S, T) of a U(4)/SU(4) irrep with u4lib (C++).
---

# Physical (spin-isospin) basis enumeration (C++)

Read [shared runtime instructions](../runtime.md). Use `u4::physicalBasis`
from [U4basis.h](../../../include/U4basis.h), which returns the list of
`u4::physicalvector` states `|(P1,P2,P3) zeta S T>` of an irrep. This enumerates
the SU(2)xSU(2) = spin x isospin content; the per-(S,T) multiplicity count alone
is [inner-mult](../inner-mult/SKILL.md), and expanding a physical state in the
Gelfand basis is [phys-to-cano](../phys-to-cano/SKILL.md). These getters do not
require coefficient database initialization.

## What it returns

- `u4::physicalBasis(irrep)` lists one `physicalvector` per `(S,T)` that occurs
  with nonzero multiplicity; **(S,T) pairs with zeta=0 are omitted** (not every
  grid point appears).
- Each `physicalvector` reports `SS()` and `TT()` — the **doubled** labels
  `SS=2S`, `TT=2T` — and `zeta()` (= `zetamax()`), the number of independent
  copies of that `(S,T)`, plus `P1()/P2()/P3()` and the U(4) labels. A
  physicalvector is a reduced label set, not a single magnetic state; the
  `(2S+1)(2T+1)` projections are implicit.
- Build the irrep with `u4::irrep(P1,P2,P3)` or `u4::irrep(n14,n24,n34,n44)`.
  Construct one state directly with `u4::physicalvector(irrep, SS, TT)` and check
  `zeta()>0` before using it (remember SS, TT are doubled).

## Enumerate and check

```cpp
#include <u4lib.h>
#include <cassert>
#include <iostream>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  // Irrep (1,1,1) with dimension 64 exhibits inner multiplicity zeta = 2:
  u4::irrep ir(1, 1, 1);
  assert(ir.dim() == 64);

  auto basis = u4::physicalBasis(ir);
  long dsum = 0;
  for (const auto& v : basis) {
    long term = static_cast<long>(v.zeta()) * (v.SS() + 1) * (v.TT() + 1);
    dsum += term;
  }
  // Dimension sum rule: sum zeta*(2S+1)*(2T+1) == dim
  assert(dsum == ir.dim() && dsum == 64);

  // Specifically check (S,T) = (1,1) (doubled SS=2, TT=2) has multiplicity zeta = 2:
  assert(u4::Zeta(ir, 2, 2) == 2);

  u4::finalize();
}
```

## Presentation and checks

Return each state with its index, `(S,T)` (state whether you report doubled
`SS,TT` or `S,T`), and `zeta`. The dimension sum rule
`sum zeta*(2S+1)*(2T+1) == dim(irrep)` is the primary check; each term's `zeta`
equals `u4::Zeta(irrep,SS,TT)`. Keep omitted (S,T) omitted — a missing pair means
zero multiplicity, not an error.
