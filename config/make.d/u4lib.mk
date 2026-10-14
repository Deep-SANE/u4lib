# Include paths for u4lib's own headers, its utils forwarding layer, and u3lib.
CXXFLAGS += -I$(U4LIB_ROOT)/include
CXXFLAGS += -I$(U4LIB_ROOT)/include/internal
CXXFLAGS += -I$(U4LIB_ROOT)/utils

# u3lib dependency (set by configure.sh / config.generated.mk)
CXXFLAGS += -I$(U3LIB_DIR)/include
CXXFLAGS += -I$(U3LIB_DIR)/include/internal
CXXFLAGS += -I$(U3LIB_DIR)/utils
