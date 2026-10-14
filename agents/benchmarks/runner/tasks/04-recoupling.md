# Task 4: Complete three-irrep SU(4) change of coupling basis

Implement `consumer/recouple.py` exposing `recoupling(kind,A,B,C,T)`, where
kind is "U" or "Z" and each irrep is an SU(4) (P1,P2,P3) triple. Return
(left_channels,right_channels,matrix); matrix is an owning NumPy float64 array
with rows left and columns right. Each channel is a tuple
(P1_intermediate,P2_intermediate,P3_intermediate,eta_first,eta_final), sorted
lexicographically.

For both kinds the left route is (A B)->X, then (X C)->T.
For U the right route is (B C)->Y, then (A Y)->T.
For Z the right route is (A C)->Y, then (Y B)->T.
Return the library U or Z coefficient (u4u6db / u4z6db) for each labelled pair.
Include every allowed intermediate and all multiplicity indices, not just a
selected block.

Test inputs have a nonempty total space; all external, total and participating
intermediate labels are <=5. Discover complete channels rather than accepting an
intermediate list from the caller. No clipping of a complete space is allowed.
Own initialization/finalization; no caller setup or concurrent client is assumed.
Include a local check of row and column orthogonality, and an example with outer
multiplicity >1. Support asymmetric external irreps as well as equal ones.
