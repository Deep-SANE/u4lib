# Task 3: Python SU(2)xSU(2) physical coefficient batch adapter

Implement `consumer/physical.py` exposing `physical_rows(requests)`.
Each request is a tuple (A,SSa,TTa, B,SSb,TTb, C,SSc,TTc): three SU(4) irreps
(P1,P2,P3) each with a spin-isospin label pair given DOUBLED (SS=2S, TT=2T).
Return a list of independent owning NumPy float64 arrays, one per request,
indexed by (zetaA,zetaB,zetaC,eta) -- the three spin-isospin branching
multiplicities and the SU(4) outer multiplicity. These are SU(2)xSU(2)-reduced
physical coupling coefficients (wigU4SU2SU2db), not full magnetic
Clebsch-Gordan coefficients.

Support irreps with P1,P2,P3 <= 4. Each requested (S,T) occurs in its irrep
(Zeta > 0) and the triple satisfies the spin and isospin triangle conditions with
nonzero outer multiplicity. Include zero entries and every multiplicity index.
Returned arrays must retain their values across subsequent calls and modification
of a different returned array. Repeated calls and an empty batch must work in a
fresh process without caller initialization. The adapter owns the library
lifecycle for each call. Supply a small runnable example with a multiplicity > 1.
