# ---------------------------------------------------------------------------
# Pretty / quiet build output.
#
#   Default      : one short coloured line per action, e.g.  "  CXX   foo.o"
#   make V=1     : restore the full command lines (the old, verbose behaviour)
#
# Target paths are shown relative to the library, prefixed with the library
# name, so "../bin/foo" or "/abs/SAM/lib/libsam.a" print as "u3lib/bin/foo"
# / "SAM/lib/libsam.a" instead of a confusing ".." or a long absolute path.
#
# Usage in a per-directory Makefile (always a <library>/src or <library>/app):
#   include $(CONFIG_DIR)/make.d/pretty.mk
#   ...
#   %.o: %.cpp
#       $(call pretty_print,CXX,$@)
#       $(Q)$(CXX) $(CXXFLAGS) -c $< -o $@
# ---------------------------------------------------------------------------

# Library root = parent of this src/app directory; its basename is the label.
PRETTY_LIB_ROOT := $(abspath $(CURDIR)/..)
PRETTY_LIB_NAME := $(notdir $(PRETTY_LIB_ROOT))

# Map a target path to "<library>/<path-inside-library>" for display. Anchoring
# on the library *name* (not the full root path) keeps it robust to symlinked
# or differently-spelled roots (e.g. macOS /tmp vs /private/tmp).
pretty_tgt = $(PRETTY_LIB_NAME)/$(lastword $(subst /$(PRETTY_LIB_NAME)/, ,$(abspath $(1))))

ifeq ($(V),1)
  Q :=
  pretty_print = @:
else
  Q := @
  # $(call pretty_print,TAG,target) -> "  <bold-blue TAG>  <library>/path"
  pretty_print = @printf '  \033[1;34m%-5s\033[0m %s\n' '$(1)' '$(call pretty_tgt,$(2))'
endif
