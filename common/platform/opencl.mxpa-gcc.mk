# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# Uncomment below two lines and configure if you want to use a platform
# other than global one

# gcc (default)
CC = gcc
PLATFORM_CFLAGS = -g -O3
  
CXX = g++
PLATFORM_CXXFLAGS = -g -O3

LINKER = gcc
PLATFORM_LDFLAGS = -lmxpa_runtime -lm -lpthread -Wl,-rpath,'$$ORIGIN' -g -L$(MXPA_PATH)/lib

CCL = $(MXPA_PATH)/bin/mxpa.py
PLATFORM_CCLFLAGS = -c gcc --keep --verbose

ifeq ("$(notdir $(BUILDDIR))", "")
$(error $$BUILDDIR is not set correctly)
endif

ifneq ("$(notdir $(patsubst %/,%,$(dir $(BUILDDIR))))", "build")
$(error $$BUILDDIR is not set correctly)
endif

SHIP_PRECOMPILED_KERNELS_DIR=$(BUILDDIR)

KERNEL_OBJS=$(addprefix $(BUILDDIR)/,$(KERNEL_FILES))

.PHONY: KERNELS
KERNELS : ${KERNEL_OBJS}  

$(BUILDDIR)/%.cl : $(SRCDIR)/%.cl
	$(CCL) -o $(SHIP_PRECOMPILED_KERNELS_DIR) ${PLATFORM_CCLFLAGS} ${APP_CCLFLAGS} $< 
   
.PHONY: resolvelibOpenCL
resolvelibOpenCL :

LIBOPENCL=-lmxpa_runtime -ldl
