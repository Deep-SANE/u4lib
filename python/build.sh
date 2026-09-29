#!/usr/bin/env bash
# ============================================================================
#  Convenience wrapper around `make` for the u4lib Python extension.
#
#  The real recipe lives in the Makefile, which reuses the qcx-config make.d
#  fragments -- so NO machine-specific paths are hardcoded here. Source your
#  qcx-config environment first, then run this (or just `make -C python`):
#
#      source "$QCX_ROOT/qcx-config/env.d/<you>.env.sh"
#      ./build.sh                 # or: PYTHON=/path/to/python ./build.sh
#
#  Any arguments are forwarded to make (e.g. `./build.sh clean`).
# ============================================================================
set -euo pipefail

if [[ -z "${QCX_CONFIG_DIR:-}" ]]; then
  echo "error: QCX_CONFIG_DIR is not set." >&2
  echo "Source your qcx-config environment first, e.g.:" >&2
  echo "    source \"\$QCX_ROOT/qcx-config/env.d/<you>.env.sh\"" >&2
  exit 1
fi

exec make -C "$(cd "$(dirname "$0")" && pwd)" "$@"
