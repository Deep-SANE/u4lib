# Shared execution instructions (Python)

Use these instructions with any `skills-python/` calculation skill. Read
[README](../../README.md) for the full build and convention reference.

## Environment and execution

Each skill's main example is a standalone calculation with the shared setup
shown below. When combining skills, combine their calculation bodies inside one
session rather than repeating their lifecycle wrappers. API signatures and
mathematical formulas are reference notation, not standalone programs.

- u4lib configures itself once with `make config` (which writes
  `config/config.generated.mk`); no environment variables are sourced. It layers
  on u3lib and inherits its Wigner backend (libwignernj) and toolchain.
- Use the Python interpreter matching the extension ABI. If the extension is
  missing or stale, run `make python PYTHON=/path/to/interpreter` from the
  repository root with the intended interpreter. u4lib imports u3lib internally,
  so u3lib's `python/` must also be importable. The null-space solve uses krans
  (dense-QR fallback).
- Use the module path supplied by the environment and inspect `u4lib.__file__`.
  For a repository-local build, add its `python/` directory to the import path.
  Do not override an explicitly supplied runtime with a stale local extension.
- For a standalone calculation, use this lifecycle around coefficient work:

```python
import u4lib as u4
u4.initialize()
try:
    # Run the selected skill's calculation here.
    pass
finally:
    u4.finalize()
```

`u4.initialize()` cascades u4 → u3 (Wigner backend + SO(3) caches +
the U(3) **and** U(4) databases), and `u4.finalize()` tears down in reverse — so
within u4lib the co-use cascade is automatic. If you also drive standalone u3lib
in the same process, init both up front, finalize both at the very end, and never
finalize one mid-workflow (shared Wigner backend).

## Inputs and results

- SU(4) labels P1,P2,P3 are nonnegative integers. U(4) labels are ordered
  nonnegative integers satisfying n14>=n24>=n34>=n44. Do not silently treat a
  four-label U(4) request as a three-label SU(4) request or remove a determinant
  shift without checking its meaning.
- **(S,T) use doubled labels: `SS=2S`, `TT=2T`** (kept integer). The (S,T)
  branching multiplicity `Zeta` is valid only on the parity-correct (S,T) grid;
  off-grid it returns 0. Enumerate valid states with `physicalBasis(irrep)`
  rather than scanning `Zeta`.
- Library integer storage is bounded: inspect the API types before unusually
  large requests. The test coverage P1,P2,P3<=5 is not a general API limit.
- Validate labels/indices before the call: invalid ones can abort the process.
  Do not disable assertions as an input-validation strategy.
- Multiplicity maxima are counts; valid indices are 0 through count-1. Label
  output axes explicitly, including whether the presentation is zero-based.
- Use the library's fixed basis, signs, and normalization. Do not rotate a
  multiplicity basis or align signs merely to match another convention.
- For a forbidden mathematical channel, report that fact without indexing an
  empty result. A zero entry in an allowed block is a valid coefficient.
- Return the requested numbers with labels, a reproducible script, and the check
  actually performed. Do not claim a calculation succeeded without executing it.
  Report missing dependencies if execution cannot proceed.

Use [documented references](../../test/reference/README.md) for signed regression
checks. Numerical agreement is tolerance-based, not guaranteed bitwise identity
across backends. If modifying bindings, rebuild/import with the same Python.
Never regenerate references just to make a disagreement disappear.
