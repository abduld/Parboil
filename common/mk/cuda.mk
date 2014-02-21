# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

# CUDA specific
LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include -I$(CUDA_PATH)/include
LANG_CXXFLAGS=$(LANG_CFLAGS)
LANG_LDFLAGS=-L$(CUDA_LIB_PATH)

LANG_CUDACFLAGS=$(LANG_CFLAGS)

# Note, OPT_CFLAGS is used for C, C++, and CUDA

CFLAGS=$(OPT_CFLAGS) $(APP_CFLAGS) $(LANG_CFLAGS) $(PLATFORM_CFLAGS)
CXXFLAGS=$(OPT_CFLAGS) $(APP_CXXFLAGS) $(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS)

CUDACFLAGS=$(OPT_CFLAGS) $(LANG_CUDACFLAGS) $(PLATFORM_CUDACFLAGS) $(APP_CUDACFLAGS) 
CUDALDFLAGS=$(LANG_LDFLAGS) $(PLATFORM_CUDALDFLAGS) $(APP_CUDALDFLAGS)

# Rules common to all makefiles

########################################
# Functions
########################################

# Add BUILDDIR as a prefix to each element of $1
INBUILDDIR=$(addprefix $(BUILDDIR)/,$(1))

# Add SRCDIR as a prefix to each element of $1
INSRCDIR=$(addprefix $(SRCDIR)/,$(1))


########################################
# Environment variable check
########################################

# The second-last directory in the $(BUILDDIR) path
# must have the name "build".  This reduces the risk of terrible
# accidents if paths are not set up correctly.
ifeq ("$(notdir $(BUILDDIR))", "")
$(error $$BUILDDIR is not set correctly)
endif

ifneq ("$(notdir $(patsubst %/,%,$(dir $(BUILDDIR))))", "build")
$(error $$BUILDDIR is not set correctly)
endif

.PHONY: run

ifeq ($(CUDA_PATH),)
FAILSAFE=no_cuda
else 
FAILSAFE=
endif

########################################
# Derived variables
########################################

ifeq ($(DEBUGGER),)
DEBUGGER=gdb
endif

OBJS = $(call INBUILDDIR,$(SRCDIR_OBJS))

ifeq ($(OPTIMIZATION),0)
OPT_CFLAGS=-O0
OPT_LDFLAGS=
endif
ifeq ($(OPTIMIZATION),1)
OPT_CFLAGS=-O1
OPT_LDFLAGS=
endif
ifeq ($(OPTIMIZATION),2)
OPT_CFLAGS=-O2
OPT_LDFLAGS=
endif
ifeq ($(OPTIMIZATION),3)
OPT_CFLAGS=-O3
OPT_LDFLAGS=
endif

########################################
# Rules
########################################

default: $(FAILSAFE) $(BUILDDIR) $(BIN)

run:
	@echo "Resolving CUDA runtime library..."
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ldd $(BIN) | grep cuda
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) $(BIN) $(ARGS)

debug:
	@echo "Resolving CUDA runtime library..."
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ldd $(BIN) | grep cuda
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) $(DEBUGGER) --args $(BIN) $(ARGS)

clean :
	rm -rf $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi

$(BIN) : $(OBJS) $(BUILDDIR)/parboil_cuda.o $(BUILDDIR)/args.o
	$(CUDALINK) -o $@ $(CUDALDFLAGS) $^

$(BUILDDIR)/parboil_cuda.o : $(PARBOIL_ROOT)/common/src/parboil_cuda.c ${BUILDDIR}
	$(CUDACC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/parboil.o : $(PARBOIL_ROOT)/common/src/parboil.c ${BUILDDIR}
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/args.o : $(PARBOIL_ROOT)/common/src/args.c ${BUILDDIR}
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	$(CUDACC) $< $(CUDACFLAGS) -c -o $@

no_cuda:
	@echo "CUDA_PATH is not set. Open $(CUDA_ROOT)/common/Makefile.conf to set default value."
	@echo "You may use $(PLATFORM_MK) if you want a platform specific configurations."
	@exit 1

