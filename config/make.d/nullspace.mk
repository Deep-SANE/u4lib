# ---------------------------------------------------------------------------
# Nullspace-solver backend selection.
#
#   NULLSPACE_SOLVER = qr     (default) -> dense QR         (always available)
#                      krans              -> null_space_krans   (requires USE_KRANS=1;
#                                          its wrapper falls back to dense QR)
#
# Emitted macros (consumed by the callers):
#   -DNULLSPACE_SOLVER_KRANS   select the krans include + solve call.
#   -DNULLSPACE_SPARSE       build the projection matrix sparse (krans); when
#                            absent the matrix is built dense for the QR solver.
# ---------------------------------------------------------------------------
NULLSPACE_SOLVER ?= qr

ifeq ($(NULLSPACE_SOLVER),qr)
  # dense QR; no macro needed (callers default to QR)
else ifeq ($(NULLSPACE_SOLVER),krans)
  ifneq ($(USE_KRANS),1)
    $(error NULLSPACE_SOLVER=krans requires USE_KRANS=1)
  endif
  CXXFLAGS += -DNULLSPACE_SOLVER_KRANS -DNULLSPACE_SPARSE
else
  $(error Invalid NULLSPACE_SOLVER='$(NULLSPACE_SOLVER)'; choose qr or krans)
endif
