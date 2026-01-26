include $(QCX_CONFIG_DIR)/config.mk

export 

.PHONY: all
all: 
	$(MAKE) -C src

.PHONY: test
test:
	$(MAKE) -C src/test

.PHONY: python
python:
	$(MAKE) -C python

.PHONY: clean
clean:
	$(MAKE) -C src clean
	$(MAKE) -C src/test clean
	$(MAKE) -C python clean
	
