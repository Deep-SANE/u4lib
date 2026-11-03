# libkrans (krans) sparse null-space solvers. Active when USE_KRANS=1.
ifeq ($(USE_KRANS),1)

CXXFLAGS += -DUSE_KRANS

ifdef KRANS_DIR
CXXFLAGS += -I$(KRANS_DIR)/include
LDFLAGS += -L$(KRANS_DIR)/build
endif

# Static libs; krans_eigen depends on krans_core, so it must come first.
LDFLAGS += -lkrans_eigen -lkrans_core

endif
