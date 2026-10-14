# Task 2: u9recoupling — C++ batch exporter for four-irrep SU(4) 9-coefficient blocks

Build `consumer/u9_batch`. Read requests from stdin until EOF. Each line has
27 integers: the SU(4) (P1,P2,P3) triples for A B C D AB CD AC BD T, in that
order. All labels are <=5 and all six coupling multiplicities are positive.

For each request q (zero-based), print `block q etaAB etaCD etaLeft etaAC etaBD etaRight`.
Here etaLeft is the multiplicity AB x CD -> T and etaRight is AC x BD -> T.
Then print `entry q iAB iCD iLeft iAC iBD iRight value` for every combination,
in lexicographic index order. The value is the library SU(4) 9-coefficient
between ((A B)AB (C D)CD)T and ((A C)AC (B D)BD)T for these indices.

The consumer needs all blocks retained in memory before serialization:
perform all coefficient requests first, and only then write the blocks.
This ordering is required for later downstream batch processing. Values must
survive subsequent requests, including requests with different block dimensions.
An empty stream produces no output and exits successfully. Supply a local batch
test with multiple distinct requests and outer multiplicity >1.
