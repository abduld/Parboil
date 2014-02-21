# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

# MPI_PATH=

CC=$(MPI_PATH)/bin/mpicc
PLATFORM_CFLAGS=-I$(MPI_PATH)/include
CXX=$(MPI_PATH)/bin/mpic++
PLATFORM_CXXFLAGS=$(PLATFORM_CFLAGS)
LINKER=$(MPI_PATH)/bin/mpic++
PLATFORM_LDFLAGS=-L$(MPI_PATH)/lib64

