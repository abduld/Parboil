# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# gcc (default)
CC = icc
PLATFORM_CFLAGS = -fopenmp
  
CXX = icpc
PLATFORM_CXXFLAGS = 
  
LINKER = icpc
PLATFORM_LDFLAGS = -lm -lpthread -ldl -liomp5 -ltbbmalloc -L$(OPENCL_LIB_PATH) -L${INTEL_COMPOSER_PATH}/tbb/lib/intel64

