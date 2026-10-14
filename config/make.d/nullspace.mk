# ---------------------------------------------------------------------------
# Nullspace-solver backend selection.
#
#   NULLSPACE_SOLVER = qr     (default) -> dense QR         (always available)
#                      sns              -> null_space_sns   (requires USE_SNS=1;
#                                          its wrapper falls back to dense QR)
#
# Emitted macros (consumed by the callers):
#   -DNULLSPACE_SOLVER_SNS   select the sns include + solve call.
#   -DNULLSPACE_SPARSE       build the projection matrix sparse (sns); when
#                            absent the matrix is built dense for the QR solver.
# ---------------------------------------------------------------------------
NULLSPACE_SOLVER ?= qr

ifeq ($(NULLSPACE_SOLVER),qr)
  # dense QR; no macro needed (callers default to QR)
else ifeq ($(NULLSPACE_SOLVER),sns)
  ifneq ($(USE_SNS),1)
    $(error NULLSPACE_SOLVER=sns requires USE_SNS=1)
  endif
  CXXFLAGS += -DNULLSPACE_SOLVER_SNS -DNULLSPACE_SPARSE
else
  $(error Invalid NULLSPACE_SOLVER='$(NULLSPACE_SOLVER)'; choose qr or sns)
endif
