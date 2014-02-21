# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# gcc (default)
CC = gcc
PLATFORM_CFLAGS = 
  
CXX = g++
PLATFORM_CXXFLAGS = 
  
LINKER = g++
PLATFORM_LDFLAGS = -lm -lpthread

.PHONY: KERNELS
KERNELS :

resolvelibOpenCL:
	@echo "Resolving OpenCL library..."
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(OPENCL_LIB_PATH) ldd $(BIN) | grep OpenCL

