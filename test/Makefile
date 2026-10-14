# The root Makefile exports the configured compiler, flags and solver selection.
BIN := ../bin/test
SOURCES := $(sort $(wildcard test_*.cpp))
PROGRAMS := $(patsubst %.cpp,$(BIN)/%,$(SOURCES))
# Prefer this checkout, and keep checks enabled even in release builds.
TEST_FLAGS = -I../include $(CXXFLAGS) -UNDEBUG

.PHONY: all test clean FORCE reference-tool
all: test
test: $(PROGRAMS)
	@echo "Running u4lib tests (solver: $(NULLSPACE_SOLVER)); signed reference comparisons, no coefficient dumps"
	@failed=0; for program in $(PROGRAMS); do \
	  echo "=== $$program ==="; \
	  if "$$program"; then :; else failed=1; fi; \
	done; \
	if [ $$failed -eq 0 ]; then echo "PASS: all test executables"; else echo "FAIL: one or more test executables"; fi; \
	exit $$failed

# FORCE intentionally rebuilds EVERY executable on EVERY make test invocation.
$(BIN)/test_%: test_%.cpp FORCE | $(BIN)
	@echo "CXX $<"
	@$(CXX) $(TEST_FLAGS) -o $@ $< $(LDFLAGS)

reference-tool: $(BIN)/generate_references
$(BIN)/generate_references: tools/generate_references.cpp $(wildcard *.h) FORCE | $(BIN)
	@echo "CXX $< (reference generator; not run by make test)"
	@$(CXX) $(TEST_FLAGS) -o $@ $< $(LDFLAGS)
$(BIN):
	@mkdir -p $@
FORCE:
clean:
	rm -f $(PROGRAMS) $(BIN)/generate_references
