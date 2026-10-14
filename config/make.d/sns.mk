# libsns (dp-sparse-nullspace) sparse null-space solvers. Active when USE_SNS=1.
ifeq ($(USE_SNS),1)

CXXFLAGS += -DUSE_SNS

ifdef SNS_DIR
CXXFLAGS += -I$(SNS_DIR)/include
LDFLAGS += -L$(SNS_DIR)/lib
endif

# Static libs; sns_eigen depends on sns_core, so it must come first.
LDFLAGS += -lsns_eigen -lsns_core

endif
