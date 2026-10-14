---
name: conjugation
description: Compute Gelfand or physical-state conjugation for SU(4), including self-conjugate zeta mixing.
---

# conjugation

Read [shared runtime instructions](../runtime.md) before execution. The C++
examples below assume the `u4::` API and the lifecycle described there.

The SU(4) conjugate of `(P1,P2,P3)` is `(P3,P2,P1)`; a self-conjugate irrep has
`P1=P3`. For irreps use `irrep.conjugate()` and `irrep.conjugate_phase_power()`
(the conjugation sign is `(-1)**power`). For Gelfand states use
`g.conjugate()` and `g.conjugate_phase()`; the action is
`C|g> = (-1)^(n14+n34-n13-n23-n33-n12-n22-n11) |g.conjugate()>`. For complex
superpositions, conjugate the amplitudes as well as mapping the basis states.

```cpp
u4::irrep c = u4::irrep(2, 0, 0).conjugate();
assert(c.P1() == 0 && c.P2() == 0 && c.P3() == 2);    // (P1,P2,P3) -> (P3,P2,P1)
assert(u4::irrep(2, 0, 0).conjugate_phase_power() == 3);
auto g = u4::gelfandBasis(u4::irrep(2, 0, 0))[1];
assert(g.conjugate().check());                        // conjugate is a valid pattern
assert(g.conjugate().conjugate() == g);               // C^2 = identity on the state
```

For the physical spin-isospin states, in the configured phase convention with
`R=(P1,P2,P3)`, `barR=(P3,P2,P1)` (labels doubled, `SS=2S`, `TT=2T`):

- distinct irreps (`P1 != P3`):
  `C|R zeta S MS T MT> = eps_R (-1)^(S-MS+T-MT) |barR zeta S,-MS T,-MT>`,
  with `eps_R = (-1)^(P1+P2+P3)` for `P1>P3` and `(-1)^P2` for `P1<P3`;
- self-conjugate irreps (`P1 = P3`): conjugation acts as a projection-independent
  symmetric matrix `D` with `D*D = I` that may **mix** zeta (e.g. `R=(1,1,1)`,
  `S=T=1`) — not a scalar phase.

To build `D` or apply the general operator, use the `phys_to_cano` /
`phys_to_cano_allM` transforms (see [phys-to-cano](../phys-to-cano/SKILL.md)),
apply the signed Gelfand permutation,
then project onto the conjugate physical basis. Read the
[transformation header](../../../include/U4SU2SU2_to_U4U3U2U1.h) and
[conjugation checks](../../../test/conjugation_cases.h). Check reconstruction,
norm, and double conjugation; do not diagonalize `D` unless the user asks for a
different basis.
