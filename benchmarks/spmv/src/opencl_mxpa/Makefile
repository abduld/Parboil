# (c) 2010 The Board of Trustees of the University of Illinois.

LANGUAGE=opencl
TOOLS_SRC=common_src/convert-dataset
SRCDIR_OBJS=main.o file.o ocl.o
APP_CFLAGS=-ffast-math -g3 -O3 -I$(TOOLS_SRC)
APP_CXXFLAGS=-ffast-math -g3 -O3 -I$(TOOLS_SRC)
KERNEL_FILES=kernel.cl

include $(TOOLS_SRC)/commontools.mk

