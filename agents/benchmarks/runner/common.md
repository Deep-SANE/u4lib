You are integrating the supplied u4lib into an external scientific consumer.
Implement the requested reusable component in consumer/. You may read the supplied
library source, documentation, tutorials and tests. Do not modify the library or
its reference data. Use its numerical routines; do not hardcode coefficient
answers or implement a replacement solver. No network access is available.

Inspect the supplied README and any repository-local agent guidance available
in this snapshot. Use that guidance when provided; it may contain skills relevant
to the task. The runner supplies a fixed environment script, GNU C++, Python,
NumPy and required dependencies. The matching Python extensions (u4lib, and the
u3lib it layers on) are already built. Source `./environment.sh` from the
workspace root before building or importing. Do not hardcode machine paths.
Deliver source, local tests, and a short README with build/use instructions.
C++ submissions must build with `make -C consumer` and create the named executable.
Python submissions must import from consumer with the supplied interpreter.
You may run your own tests. Your final text is not the submitted software.

All irreps are SU(4) (P1,P2,P3), normalized to U(4) [P1+P2+P3,P2+P3,P3,0] when
needed. Spin and isospin are stored DOUBLED: SS = 2S and TT = 2T (likewise
MMS = 2*MS, MMT = 2*MT). All multiplicity indices in consumer interfaces are
zero-based. Retain the library's basis order, phases, and normalization. Unless a
task says otherwise, inputs are valid within the stated domain. Integer and label
outputs must match exactly. Floating values use tolerance 1e-12 + 1e-10*abs(expected);
matrix identity residuals use Frobenius tolerance 1e-9. Emit no NaN or infinity.
For command-line programs send diagnostics to stderr, results only to stdout.
