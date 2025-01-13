include ../config.mk

BIN = ../bin

SRCS = $(wildcard *.cpp)
PRGS = $(addprefix $(BIN)/,$(basename $(SRCS)))

CXXFLAGS += -I../include

#LDFLAGS := -L../src -lSU3 $(LDFLAGS)

# debug messages:
#$(info $$SRCS is [$(SRCS)])
#$(info $$PRGS is [$(PRGS)])

.PHONY: all
all: $(PRGS)

$(BIN)/%: %.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(PRGS)
