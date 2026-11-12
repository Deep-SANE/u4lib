---
name: install
description: Build u4lib from a fresh checkout (make config -> make -> make test) before any calculation skill.
---

# install

Run this once, from the u4lib root, before any calculation skill, when the
library is not built yet. First detect the built state: if
`config/config.generated.mk` exists **and** `bin/` (for C++) or
`python/u4lib.so` (for Python) is present, u4lib is already configured — skip to
the calculation skills. u4lib needs no manual environment variables; `make config`
resolves every dependency (inheriting from `u3lib`) and records the result in
`config/config.generated.mk`.

## Prerequisites (system tools; the library does not install these)

- A C++20 compiler with OpenMP (real GCC, or Clang with libomp; **macOS note**: default Apple Clang `/usr/bin/clang++` lacks native OpenMP, install GCC via `brew install gcc` and export `CXX=g++-14`, or use Homebrew LLVM with `libomp`).
- `u3lib` built and configured (e.g. in `../u3lib` or specified via `U3LIB_DIR`).
- For the Python extension only: a Python interpreter with `pybind11` (and `ipykernel`
  for the tutorial). `make python` auto-detects a suitable one, or pass `PYTHON=`.

## 1. Configure

`make config` detects `u3lib` and inherits its toolchain, Eigen3, Boost, fmt, and libwignernj backend:

- Interactive: `make config` — press Enter at each prompt to accept a detected
  path.
- Non-interactive: `make config CONFIG_ARGS="--yes"` — detect, or pass an explicit
  path: `make config CONFIG_ARGS="--u3lib=/path/to/u3lib --yes"`.

The null-space solver defaults to **KraNS** (`krans`), inherited from `u3lib`, with an automatic dense-QR fallback for small systems (override with `--solver=qr` if an external sparse solver is not desired).

## 2. Build and verify

- `make -j` — build the C++ driver programs in parallel into `bin/`.
- `make test` — build and run the regression suite. Success prints
  `PASS: all test executables`. Treat any `FAIL` as a blocking error: report it
  and do not proceed to calculations.

## 3. Python extension (only for a Python calculation)

- `make python` — build `python/u4lib.so`. The interpreter is auto-detected (one
  with `pybind11`, preferring one that also has `ipykernel` for the Jupyter
  tutorial); override with `make python PYTHON=/path/to/python`. Verify the import
  and a known value:

  ```sh
  cd python && python3 -c "import u4lib as u; u.initialize(); print(u.irrep(1,1,1).dim()); u.finalize()"
  ```

  should print `64`.

## Troubleshooting

- `No configuration found. Run make config first.` — the generated config is
  missing; do step 1.
- `u3lib not found`: ensure `u3lib` is configured and built in `../u3lib` or provide
  `--u3lib=/path/to/u3lib`.
- To reset to a clean checkout and start over: `make distclean`.
- Do not hand-edit `config/config.generated.mk` or disable assertions; re-run
  `make config` to change the configuration.
