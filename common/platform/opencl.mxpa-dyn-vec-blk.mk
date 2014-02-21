# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# Uncomment below two lines and configure if you want to use a platform
# other than global one

PLATFORM_NAME=-

# gcc (default)
CC = $(INTEL_COMPOSER_PATH)/bin/intel64/icc
PLATFORM_CFLAGS = -D__MXPA__ -g -O3 -xHOST
  
CXX = $(INTEL_COMPOSER_PATH)/bin/intel64/icpc
PLATFORM_CXXFLAGS = -D__MXPA__ -g -O3 -xHOST

LINKER = $(INTEL_COMPOSER_PATH)/bin/intel64/icc
PLATFORM_LDFLAGS = -lm -ltbb -lpthread -lrt -Wl,-rpath,'$$ORIGIN:${INTEL_COMPOSER_PATH}/compiler/lib/intel64:${INTEL_COMPOSER_PATH}/tbb/lib/intel64' -g

CCL = $(MXPA_PATH)/bin/mxpa.py
PLATFORM_CCLFLAGS = -c mxpa_dyn_vec_blk --keep --debug --verbose

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
	@echo "$(CCL) -o $(SHIP_PRECOMPILED_KERNELS_DIR) ${PLATFORM_CCLFLAGS} ${APP_CCLFLAGS} $< "
	$(CCL) -o $(SHIP_PRECOMPILED_KERNELS_DIR) ${PLATFORM_CCLFLAGS} ${APP_CCLFLAGS} $< 
   
.PHONY: resolvelibOpenCL
resolvelibOpenCL :

LIBOPENCL=-lmxpa_runtime -ldl -ltbbmalloc -L$(OPENCL_LIB_PATH) -L${INTEL_COMPOSER_PATH}/tbb/lib/intel64 -L$(MXPA_PATH)/lib
