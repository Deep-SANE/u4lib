# u4lib Python bindings — build notes

`pyu4lib.cpp` builds into `u4lib.so` (a pybind11 extension) — Wigner-Racah
algebra for U(4)/SU(4). Tested on macOS/arm64 with the GNU toolchain (Homebrew
gcc-16) + conda Python 3.14. (See also `u3lib/python/BUILD_NOTES.md`; u4lib
layers on u3lib and follows the same conventions.)

## Building
No dependency paths are hardcoded — they come from the **qcx-config
environment**, like the C++ build. Source your env first, then from the u4lib
root:

```
source "$QCX_ROOT/qcx-config/env.d/<you>.env.sh"
make python          # or: make -C python   (./build.sh is a wrapper for this)
```

Plain `make` builds only `src/`; the Python module is the separate `make python`
target. The `python/Makefile` reuses the qcx-config `make.d` fragments and pulls
in the u4lib + u3lib + qcx-utils headers (u4lib is header-only). It resets the
app `CXXFLAGS`/`LDFLAGS` the top-level Makefile exports, so the `.so` carries no
MPI/SLEPc/sns/libsam — only libgomp plus static wigxjpf. Build with `g++`
(`$OMPI_CXX`), OpenMP from gcc (native `-fopenmp`); override `CXX`/`PYTHON` on
the make line. Header edits trigger a rebuild (tracked via `u4lib.d`).

## What the module exposes (names mirror the C++ API 1:1)
- `initialize()` / `finalize()` — **always call these**; `initialize()` cascades
  u4 → u3 → qcx_utils (Wigner backend + SO3 caches + U(3) and U(4) databases).
  `is_initialized()` reports the state.
- Irreps `su4.irrep`/`irrep` (full ordering + hashable), `gelfand`,
  `Eta` (coupling multiplicity), `Zeta` ((S,T) branching), `physicalvector`,
  `physicalBasis`/`gelfandBasis`.
- Databases (the only way to reach coefficients): `wigU4U3db()`,
  `wigU4SU2SU2db()`, `u4u6db()`, `u4z6db()`, `u4u9db()`.
- Each database has a structured `get_U4*` accessor returning the Definitions.h
  container (`U4U3`/`U4SU2SU2`/`U4U6`/`U4Z6`/`U4U9`): multiplicity-indexed
  `__call__`, flat `.data()` numpy, and a `print()` that matches the C++
  `operator<<`. Flat numpy `get()` is kept where applicable.

## Safety guards
- **Database access** before `initialize()` / after `finalize()` raises a Python
  `RuntimeError` instead of segfaulting the kernel (per-module `require_init()`).
- **U4U9 lifetime**: `get_U4U9` returns a `std::span` into a thread-local scratch
  buffer; the binding copies it into an owning `U4U9`, so a saved object stays
  valid across later DB calls.
- **Zeta domain**: `Zeta` is only valid on the parity-correct (S,T) grid for the
  irrep (the C++ function is guarded to return 0 off-grid). Enumerate valid
  states with `physicalBasis(irrep)` rather than scanning `Zeta`.
- **Asserts are on** (no `-DNDEBUG`): out-of-range container indices `abort()`
  the process (an empty/forbidden coupling returns an empty container, which is
  safe to query for shape but not to index).

## Using u3lib and u4lib together
u3lib and u4lib are separate extension modules that **share one wigxjpf backend**
in the process. `initialize()`/`finalize()` are idempotent per module, but those
flags do not coordinate the shared backend across modules — so finalizing one
library while the other is still in use crashes the process.

Safe recipe when using both (e.g. an ENN precompute needing U(3) and U(4)
coefficients): initialize both up front, do all the work, finalize both at the
end.

```python
import u3lib as u3, u4lib as u4
u3.initialize(); u4.initialize()
#  ... all u3 and u4 coefficient work ...
u4.finalize(); u3.finalize()     # at the very end
```

Do **not** call one library's `finalize()` mid-workflow while the other is still
in use. (Within u4lib this is automatic: `u4.finalize()` already cascades to u3
and qcx_utils.) The `require_init()` guard is per-module, so it cannot detect
this cross-module case.
