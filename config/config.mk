# u4lib build configuration — self-contained, no environment variables.
#
# Paths are resolved from this file's own location, and the resolved
# dependency settings come from config.generated.mk, which `make config`
# writes (inheriting toolchain and core library settings from u3lib).
# Nothing here is read from the environment.

# This file lives in <root>/config/, so the library root is one level up.
CONFIG_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
U4LIB_ROOT := $(abspath $(CONFIG_DIR)/..)

ifeq ($(wildcard $(CONFIG_DIR)/config.generated.mk),)
$(error No configuration found. Run `make config` first.)
endif
include $(CONFIG_DIR)/config.generated.mk

# Include paths for u4lib and u3lib dependency.
include $(CONFIG_DIR)/make.d/u4lib.mk

# Dependency fragments (consume the variables set in config.generated.mk).
include $(CONFIG_DIR)/make.d/boost.mk
include $(CONFIG_DIR)/make.d/eigen.mk
include $(CONFIG_DIR)/make.d/fmt.mk
include $(CONFIG_DIR)/make.d/openmp.mk
include $(CONFIG_DIR)/make.d/wignernj.mk
include $(CONFIG_DIR)/make.d/sns.mk
include $(CONFIG_DIR)/make.d/nullspace.mk
include $(CONFIG_DIR)/make.d/pretty.mk
