# Task 2: Python SU(4)>U(3) canonical coefficient batch adapter

Implement `consumer/canonical.py` exposing `canonical_rows(requests)`.
Each request is a tuple (a,b,c); each state is seven integers
(n14,n24,n34,n44,n13,n23,n33): the U(4) top row [n14,n24,n34,n44] and its U(3)
sublabels [n13,n23,n33], with normalized top row and valid interlacing. Return a list of independent owning NumPy float64 arrays,
one per request, indexed by (rho, eta) -- the U(3) inner multiplicity rho and the
SU(4) outer multiplicity eta. These are SU(4)>U(3) reduced coefficients, not full
magnetic Clebsch-Gordan coefficients.

Support input and output irreps with P1,P2,P3 <= 4. Requests satisfy the subgroup
coupling selection rules and have nonzero outer multiplicity. Include zero
coefficient entries and every (rho,eta). Returned arrays must retain their values
across subsequent calls and modification of a different returned array. Repeated
calls and an empty batch must work in a fresh process without caller
initialization. The adapter owns the library lifecycle for each call; no other
library client is active. Supply a small runnable example with a multiplicity > 1.
