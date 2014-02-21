# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# gcc (default)
CC = icc
PLATFORM_CFLAGS = 
  
CXX = icpc
PLATFORM_CXXFLAGS = 
  
LINKER = icpc
PLATFORM_LDFLAGS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread

