# We requires both of the following paths to be set upon including this makefile
# TEST_VERIFIABLE_SRCDIR = <points to this directory>
# TEST_VERIFIABLE_BUILDDIR = <points to destination of .o file>

TEST_VERIFIABLE_HDRS = $(TEST_VERIFIABLE_SRCDIR)/verifiable.h
TEST_VERIFIABLE_OBJS = $(TEST_VERIFIABLE_BUILDDIR)/verifiable.o

$(TEST_VERIFIABLE_BUILDDIR)/verifiable.o: $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu $(TEST_VERIFY_REDUCE_HDRS)
	@printf "Compiling %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	$(NVCC) -o $@ $(NVCUFLAGS) -c $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu
