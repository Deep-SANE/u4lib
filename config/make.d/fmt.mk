ifdef HAVE_FMT

CXXFLAGS += -DHAVE_FMT

ifdef FMT_DIR
CXXFLAGS += -I$(FMT_DIR)/include
endif

ifeq ($(FMT_HEADER_ONLY),1)
# header only mode:

CXXFLAGS += -DFMT_HEADER_ONLY

else
# linked mode:

ifdef FMT_DIR
LDFLAGS += -L$(FMT_DIR)/lib
endif

LDFLAGS += -lfmt

endif

endif
