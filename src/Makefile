include $(CONFIG_DIR)/make.d/pretty.mk

BIN = ../bin

SRCS = $(wildcard *.cpp)
PRGS = $(addprefix $(BIN)/,$(basename $(SRCS)))

.PHONY: all
all: $(PRGS)

$(BIN)/%: %.cpp | $(BIN)
	$(call pretty_print,CXX,$@)
	$(Q)$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(PRGS)
