ifeq ($(USE_OPENMP),1)

CXXFLAGS += -DUSE_OPENMP

CXXFLAGS += $(OMP_CXXFLAGS)
LDFLAGS += $(OMP_LDFLAGS)

ifdef LLVM_ROOT
CXXFLAGS += -I$(LLVM_ROOT)/include
LDFLAGS += -L$(LLVM_ROOT)/lib
OMPI_CXX = ${LLVM_ROOT}/bin/clang++
endif

ifdef OMP_ROOT
CXXFLAGS += -I$(OMP_ROOT)/include
LDFLAGS += -L$(OMP_ROOT)/lib
endif

endif
