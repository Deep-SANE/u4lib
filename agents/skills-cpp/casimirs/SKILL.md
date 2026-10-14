---
name: casimirs
description: Compute u4lib Casimir eigenvalues C1, C2, C3 and C4 with explicit U(4) labels and SU(4) normalization.
---

# Casimir eigenvalues

Read [shared runtime instructions](../runtime.md). Use the irrep methods in
[U4irrep.h](../../../include/U4irrep.h); these return eigenvalues, not matrices.

## Which Casimir applies to which irrep

- **`C2()`, `C3()` and `C4()` apply to BOTH** `su4::irrep(P1,P2,P3)` and
  `u4::irrep(...)` (the U(4) class inherits them). These are the SU(4) Casimirs
  (orders 2, 3, 4) and are **invariant under a determinant shift** (a uniform
  `n44` offset).
- **`C1()` applies ONLY to `u4::irrep`** (U(4)). SU(4) is U(4) with the U(1)
  trace generator (`E11+E22+E33+E44`) **removed from the algebra**; `C1` is the
  Casimir of exactly that removed U(1) — the total number of quanta
  `n14+n24+n34+n44` — so it is **not an SU(4) invariant at all**. `su4::irrep`
  therefore has no `C1()` method: a bare `su4::irrep(P1,P2,P3).C1()` is a category
  error, not a missing feature. Do not offer or fabricate a "C1 of an SU(4)
  irrep". C1 is defined only once a U(4) irrep is fixed — if the user gives U(4)
  labels `[n14,n24,n34,n44]` (or a total number of quanta A), compute
  `u4::irrep(...).C1()` from that; if they give only SU(4) labels `(P1,P2,P3)` and
  ask for C1, the request is under-specified — say C1 is not defined for SU(4)
  and ask for the quanta/shift rather than silently choosing a representative.

SU(4) has rank 3, so there are three independent non-linear Casimirs (C2, C3,
C4) plus the U(1) linear C1 — four in total. Present C2, C3, C4 as the SU(4)
Casimirs, not full U(4) invariants including the U(1) contribution; state the
normalization when comparing with another source.

## Calculate and check

```cpp
su4::irrep s(1, 0, 1);                          // adjoint (self-conjugate)
assert(std::abs(s.C2() - 16.0) < 1e-9);
assert(std::abs(s.C3() -  0.0) < 1e-9);         // P1 = P3 -> C3 = 0
assert(std::abs(s.C4() - 72.0) < 1e-9);
// su4::irrep has NO C1(): writing s.C1() is a compile error, not a runtime one.
u4::irrep u(1, 0, 1);                           // normalized rep [2,1,1,0]
assert(u.C1() == 4);                            // C1 = total quanta
u4::irrep shifted(3, 2, 2, 1);                  // same (P1,P2,P3)=(1,0,1), n44=1
assert(shifted.C1() == 8);                      // 4 + 4*1; C2, C3, C4 unchanged
su4::irrep a(2, 0, 1), b(1, 0, 2);              // conjugate pair
assert(std::abs(b.C3() + a.C3()) < 1e-9);       // conjugation reverses C3
```

These algebraic irrep getters do not require coefficient database
initialization. Return a labelled row with the input irrep, whether it is SU(4)
`(P1,P2,P3)` or U(4) `[n14,n24,n34,n44]`, the chosen representative when a shift
was assumed, and the requested eigenvalues; for a batch preserve input order. Do
not call `C1()` on an `su4::irrep`.
