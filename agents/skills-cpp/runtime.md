# Shared execution instructions (C++)

Use these instructions with any `skills-cpp/` calculation skill. Read
[README](../../README.md) for build setup and conventions.

## Environment and execution

Each skill's main example is a standalone calculation with the shared setup
shown below. When combining skills, combine their calculation bodies inside one
session rather than repeating their lifecycle wrappers. API signatures and
mathematical formulas are reference notation, not standalone programs.

- u4lib configures itself once with `make config` (which writes
  `config/config.generated.mk`); no environment variables are sourced. It layers
  on u3lib and inherits its Wigner backend (libwignernj) and toolchain.
- u4lib is header-only. Compile a driver (a program in `src/`, or a standalone
  `.cpp`) against the library's configured flags: the include paths for
  `u4lib/include`, `u4lib/include/internal`, `u4lib/utils`, `u3lib/include`,
  `u3lib/include/internal`, `u3lib/utils`, Eigen, Boost, and libwignernj, plus its
  link flags. The null-space solver defaults to `krans` (iterative sparse, with a dense-QR fallback); do not
  silently change the selected solver.
- For a standalone calculation, use this lifecycle around coefficient work:

```cpp
#include <u4lib.h>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);   // cascades u4 -> u3 (U(3)+U(4) DBs)
  // Run the selected skill's calculation here.
  u4::finalize();
}
```

`u4::initialize()` cascades u4 → u3 and `u4::finalize()` tears down in
reverse, so within u4lib the co-use cascade is automatic. If you also drive
standalone u3lib in the same process, init both up front, finalize both at the
very end, and never finalize one mid-workflow (shared Wigner backend).

## Multi-threading (OpenMP)

Per-thread cache and scratch buffers are OpenMP `threadprivate`, so threading
is supported exclusively inside an OpenMP parallel region. Compiling with
multi-threading requires `-fopenmp` (and `-DUSE_OPENMP`) under the configured
build flags. The main thread calls `u4::initialize(&argc, &argv)` once; inside
the `#pragma omp parallel` region, every thread must call `u4::initialize_thread()`
upon entry and `u4::finalize_thread()` before exit. Distribute work across
threads with `#pragma omp for` and guard shared I/O with `#pragma omp critical`:

```cpp
#include <omp.h>
#include <u4lib.h>

u4::initialize(&argc, &argv);
#pragma omp parallel
{
  u4::initialize_thread();
  int tid = omp_get_thread_num();
  #pragma omp for
  for (int i = 0; i < n_tasks; ++i) {
    auto blk = u4::wigU4SU2SU2db().get_U4SU2SU2(...);
    #pragma omp critical
    std::cout << "[Thread " << tid << "] " << blk(0, 0, 0, 0) << "\n";
  }
  u4::finalize_thread();
}
u4::finalize();
```

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
- Validate labels/indices before the call: invalid ones abort via `assert`.
  Do not disable assertions as an input-validation strategy.
- Multiplicity maxima are counts; valid indices are 0 through count-1. Label
  output axes explicitly, including whether the presentation is zero-based.
- Use the library's fixed basis, signs, and normalization. Do not rotate a
  multiplicity basis or align signs merely to match another convention.
- For a forbidden mathematical channel, report that fact without indexing an
  empty result. A zero entry in an allowed block is a valid coefficient.
- Return the requested numbers with labels, a reproducible driver, and the check
  actually performed. Do not claim a calculation succeeded without executing it.
  Report missing dependencies if execution cannot proceed.

Use [documented references](../../test/reference/README.md) for signed regression
checks. Numerical agreement is tolerance-based, not guaranteed bitwise identity
across backends. If modifying src/ or include/, run `make test`. Never regenerate
references just to make a disagreement disappear.
