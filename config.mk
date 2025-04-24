U4LIB_DIR:=$(abspath $(dir $(lastword $(MAKEFILE_LIST))))

include config.local.mk

include $(U3LIB_DIR)/config.mk

# header file inclusion path:
ifdef U4LIB_DIR
CXXFLAGS += -I$(U4LIB_DIR)/include
endif

# SLEPc related options
ifeq ($(USE_SLEPC),1)
CXXFLAGS += -DU4LIB_USE_SLEPC
CXXFLAGS += -DU4LIB_SPARSE_MAT
endif
