# Task 6: C++ OpenMP batch service for canonical and physical coupling

Build `consumer/coupling_batch`. Read requests from stdin until EOF; each line is
one request, tagged by a leading token:

- `C` — canonical SU(4)>U(3) reduced coupling. Followed by 21 integers: for each
  state A B C the seven labels `(n14,n24,n34,n44,n13,n23,n33)` (the U(4) row and
  its U(3) sub-row), with top-row ordering and valid interlacing.
- `P` — physical SU(2)xSU(2)-reduced coupling. Followed by 15 integers: for each
  state A B C the five labels `(P1,P2,P3,SS,TT)`, where `SS=2S`, `TT=2T` are the
  doubled spin/isospin labels.

Every request satisfies the subgroup/spin-isospin selection rules and has
positive multiplicities.

**Evaluate the whole batch in parallel with OpenMP.** Open one `#pragma omp
parallel` region; each worker thread calls `u4::initialize_thread()` on entry and
`u4::finalize_thread()` on exit, and the requests are distributed with
`#pragma omp for`. Collect every result into per-request storage during the
parallel region, and only then — on the main thread, in ascending request order
`q` (zero-based) — serialize the output. Parallel evaluation must not change the
output versus a serial run.

For each request `q`, in request order:

- canonical: print `canon q rhomax etamax`, then for every `(rho,eta)` in
  lexicographic index order a line `c q rho eta value`. `value` is the
  `wigU4U3db().get(...)` reduced coefficient (`rho` the U(3) inner multiplicity,
  `eta = Eta(A,B,C)` the SU(4) outer multiplicity).
- physical: print `phys q zetaAmax zetaBmax zetaCmax etamax`, then for every
  `(zetaA,zetaB,zetaC,eta)` in lexicographic index order a line
  `p q zetaA zetaB zetaC eta value`, from `wigU4SU2SU2db().get_U4SU2SU2(...)`.

Use the library's fixed basis, signs and normalization, including exact
selection-rule zeros. An empty stream produces no output and exits successfully.
The program must build under the configured OpenMP flags and run correctly for
any thread count (`OMP_NUM_THREADS`), with no data races and deterministic
output. Supply a local batch test that mixes `C` and `P` requests, includes a
case with multiplicity `> 1`, and runs under at least two threads.
