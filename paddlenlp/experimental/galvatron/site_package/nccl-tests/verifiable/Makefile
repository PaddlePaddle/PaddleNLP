include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NCCLDIR := $(BUILDDIR)
NVCUFLAGS += -I$(NCCLDIR)/include/ -I../include
DST_DIR := $(BUILDDIR)/test/verifiable

all: $(DST_DIR)/self_test $(DST_DIR)/verifiable.o

clean:
	rm -rf $(DST_DIR)

TEST_VERIFIABLE_SRCDIR := .
TEST_VERIFIABLE_BUILDDIR := $(DST_DIR)
include verifiable.mk

self_test: $(DST_DIR)/self_test

$(DST_DIR)/self_test: verifiable.cu verifiable.h
	@printf "Linking  %s\n" $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) -DSELF_TEST=1 verifiable.cu $(NVLDFLAGS)
