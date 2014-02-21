# (c) 2007 The Board of Trustees of the University of Illinois.

LANGUAGE=triolet
SRCDIR_OBJS=main.o file.o kernels.o
APP_CFLAGS=-ffast-math
APP_CXXFLAGS=-ffast-math
APP_TRIOLETFLAGS=-fno-parallelize
APP_LDFLAGS=-lstdc++

default: $(BUILDDIR) $(BIN)

$(BUILDDIR)/main.o : $(BUILDDIR)/kernels_cxx.h
