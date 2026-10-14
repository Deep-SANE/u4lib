---
name: cano-coupling
description: Compute SU(4)>U(3) reduced canonical coupling coefficients, including all rho and eta multiplicities (Python).
---

# cano-coupling

Read [shared runtime instructions](../runtime.md) before execution.

Each input state is seven labels: the U(4) row `[n14,n24,n34,n44]` and its U(3)
sub-row `[n13,n23,n33]`. Check top-row ordering `n14>=n24>=n34>=n44` and the
U(3) interlacing before any call.

**Quanta conservation:** In U(4), total quanta $N = \sum_{i=1}^4 n_{i4}$ is strictly
conserved under coupling: $N(A) + N(B) = N(C)$. For two normalized SU(4) $(1,1,1)$
irreps with $N=6$ ($[3,2,1,0]$), the coupled irrep $(1,2,1)$ must carry total quanta
$N=6+6=12$, corresponding to the determinant-shifted row $[5,4,2,1]$ (not the
normalized $[3,2,1,0]$). Similarly, U(3) sublabels must conserve quanta:
$\sum n_{i3}^A + \sum n_{i3}^B = \sum n_{i3}^C$.

`wigU4U3db().get(...)` returns SU(4)>U(3) **reduced** coefficients (not full
magnetic CGs) as a **2-D** numpy array over `(rho, eta)`: `rho` is the U(3)
inner multiplicity and `eta = Eta(A,B,C)` the SU(4) outer multiplicity.

```python
import math
import u4lib as u4

u4.initialize()
try:
    # Coupling (1,1,1) [3,2,1,0] x (1,1,1) [3,2,1,0] -> (1,2,1) [5,4,2,1]:
    # Demonstrates non-trivial U(3) inner multiplicity (rho = 2) and SU(4) outer multiplicity (eta = 4).
    # Sublabels: subA = [3,2,1] (quanta 6), subB = [3,2,0] (quanta 5), subC = [5,4,2] (quanta 11).
    arr = u4.wigU4U3db().get(3, 2, 1, 0,  3, 2, 1,
                             3, 2, 1, 0,  3, 2, 0,
                             5, 4, 2, 1,  5, 4, 2)
    assert arr.shape == (2, 4)   # rho = 2, eta = 4
    assert abs(float(arr[0, 0]) - math.sqrt(2.0 / 3.0)) < 1e-8   # sqrt(2/3) ~ 0.816497
    assert abs(float(arr[0, 1])) < 1e-12
    assert abs(float(arr[0, 2]) - 0.2381965337) < 1e-8
    assert abs(float(arr[1, 0])) < 1e-12
    assert abs(float(arr[1, 1]) - math.sqrt(5.0 / 8.0)) < 1e-8   # sqrt(5/8) ~ 0.790569
finally:
    u4.finalize()
```

Return every `(rho, eta)` entry or validate a requested index pair against the
array shape. These are reduced coefficients: the n11/magnetic projections are
absent; a full canonical CG additionally needs the subgroup U(3)>U(2)>U(1)
factor. The highest-weight null space, RREF, orthonormalization and
leading-sign choices fix the library `eta` basis — preserve it, and apply any
`eta`-basis change consistently to every coefficient carrying `eta`. Use scalar
identities or signed reference rows from
[canonical tests](../../../test/canonical_cases.h); an arbitrary reduced row need
not have unit norm by itself.
