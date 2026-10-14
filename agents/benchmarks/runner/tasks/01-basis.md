# Task 1: C++ SU(4) basis inventory

Build `consumer/basis_inventory`. Invocation: `basis_inventory P1 P2 P3`.
Support every integer 0 <= P1,P2,P3 <= 4. Print whitespace-delimited lines:
`dimension d`, followed by `st SS TT zeta` for every spin-isospin state with
nonzero multiplicity (SS=2S, TT=2T, doubled), in descending (SS,TT) order,
followed by `gelfand n14 n24 n34 n44 n13 n23 n33 n12 n22 n11` for every U(4)
Gelfand state in library order. Include every state once; omit absent (S,T).

This feeds a basis allocation routine: the spin-isospin multiplicity zeta counts
independent copies of (S,T), not its magnetic degeneracy. Add a local check that
the magnetic state count reconstructed from the branching, sum zeta*(SS+1)*(TT+1),
equals dimension. The grader will check labels and order, not just the total.
Example invocation: `basis_inventory 2 0 0`. No coefficient calculations are
required.
