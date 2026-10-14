ifeq ($(USE_WIGNERNJ),1)

CXXFLAGS += -DUSE_WIGNERNJ

ifdef WIGNERNJ_DIR
CXXFLAGS += -I$(WIGNERNJ_DIR)/include
LDFLAGS += -L$(WIGNERNJ_DIR)/lib
# libwignernj is a shared lib with an @rpath install name; bake the install
# lib dir into each binary as an rpath so it resolves at run time without
# needing (DY)LD_LIBRARY_PATH. -Wl,-rpath works for both Apple ld and GNU ld.
LDFLAGS += -Wl,-rpath,$(WIGNERNJ_DIR)/lib
endif

LDFLAGS += -lwignernj

endif

# libwignernj is the sole Wigner backend.
ifneq ($(USE_WIGNERNJ),1)
$(error Wigner backend not selected: set USE_WIGNERNJ=1 in your env file)
endif
