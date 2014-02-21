# (c) 2010 The Board of Trustees of the University of Illinois

LANGUAGE=triolet
TOOLS_SRC=common_src/convert-dataset
SRCDIR_OBJS=main.o file.o spmv.o
APP_CFLAGS=-I$(TOOLS_SRC)
APP_CXXFLAGS=-I$(TOOLS_SRC)
APP_TRIOLETFLAGS=-fno-parallelize

include $(TOOLS_SRC)/commontools.mk

$(BUILDDIR)/main.o : $(BUILDDIR)/spmv.o