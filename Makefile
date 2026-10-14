# u4lib — self-contained build. No environment variables required:
#   make config     resolve dependencies from u3lib, write config/config.generated.mk
#   make            build the C++ driver programs
#   make test       build and run the regression suite
#   make python     build the Python extension (make python PYTHON=/path/to/python)
#   make install    install headers to a prefix (PREFIX=..., else prompts)
#
# `make config` is interactive; for scripted/CI use pass flags through, e.g.
#   make config CONFIG_ARGS="--yes"

U4LIB_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CONFIG_DIR := $(U4LIB_ROOT)/config
export U4LIB_ROOT CONFIG_DIR

NOCONFIG_GOALS := config reconfigure clean distclean help

ifeq ($(filter $(NOCONFIG_GOALS),$(MAKECMDGOALS)),)
include $(CONFIG_DIR)/config.mk
endif
export

.PHONY: all
all:
	$(MAKE) -C src

.PHONY: config reconfigure
config reconfigure:
	@bash $(CONFIG_DIR)/configure.sh $(CONFIG_ARGS)

.PHONY: test
test:
	$(MAKE) -C test

.PHONY: python
python:
	$(MAKE) -C python

.PHONY: reference-tool
reference-tool:
	$(MAKE) -C test reference-tool

.PHONY: install
install: all
	@prefix="$(PREFIX)"; \
	if [ -z "$$prefix" ]; then \
	  read -r -p "Install prefix [$$HOME/.local]: " prefix </dev/tty || prefix=""; \
	  prefix="$${prefix:-$$HOME/.local}"; \
	fi; \
	echo "Installing u4lib headers to $$prefix/include"; \
	mkdir -p "$$prefix/include"; \
	cp -R $(U4LIB_ROOT)/include/. "$$prefix/include/"; \
	echo "Done. Downstream builds: add -I$$prefix/include"

.PHONY: clean
clean:
	rm -rf $(U4LIB_ROOT)/bin
	rm -f $(U4LIB_ROOT)/python/u4lib.so $(U4LIB_ROOT)/python/u4lib.d
	$(MAKE) -C src clean 2>/dev/null || true
	$(MAKE) -C test clean 2>/dev/null || true
	$(MAKE) -C python clean 2>/dev/null || true

.PHONY: distclean
distclean: clean
	rm -f $(CONFIG_DIR)/config.generated.mk
