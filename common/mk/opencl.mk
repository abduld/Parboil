# (c) 2007 The Board of Trustees of the University of Illinois.

-include $(PARBOIL_ROOT)/common/device.mk

# Default language wide options
# NOTE: tbb should NOT be linked together. It should be part of mxpa binaries.

LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include -I$(OPENCL_PATH)/include -I/usr/include/x86_64-linux-gnu/c++/4.7
LANG_CXXFLAGS=$(LANG_CFLAGS)
LANG_LDFLAGS=$(LIBOPENCL) -L$(OPENCL_LIB_PATH)

CFLAGS=$(OPT_CFLAGS) $(LANG_CFLAGS) $(PLATFORM_CFLAGS) $(APP_CFLAGS)
CXXFLAGS=$(OPT_CFLAGS) $(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS) $(APP_CXXFLAGS)
LDFLAGS=$(OPT_LDFLAGS) $(LANG_LDFLAGS) $(PLATFORM_LDFLAGS) $(APP_LDFLAGS) -lpfm

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

# PB_PLATFORM_ARG=$(PLATFORM_NAME)-$(PLATFORM_VERSION)
PB_PLATFORM_ARG=$(PLATFORM_NAME)
PB_DEVICE_ARG=-

ifneq ($(DEVICE_TYPE),)
PB_DEVICE_ARG=$(DEVICE_TYPE)
endif

ifneq ($(DEVICE_NAME),)
PB_DEVICE_ARG=$(DEVICE_NAME)
endif

ifneq ($(DEVICE_NUMBER),)
PB_DEVICE_ARG=$(DEVICE_NUMBER)
endif

ifeq ($(OPENCL_PATH),)
FAILSAFE=no_opencl
else 
FAILSAFE=
endif

########################################
# Derived variables
########################################

OBJS = $(call INBUILDDIR,$(SRCDIR_OBJS))

ifeq ($(DEBUGGER),)
DEBUGGER=gdb
endif

ifeq ($(OPTIMIZATION),0)
OPT_CFLAGS=-O0 -g
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
# System dependent
########################################

ifeq ($(HOST_PLATFORM),cygwin)
RUN_CMD=$(LAUNCHER) $(BIN) --platform $(PB_PLATFORM_ARG) --device $(PB_DEVICE_ARG) $(ARGS)
else
RUN_CMD=$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(OPENCL_LIB_PATH) $(LAUNCHER) $(BIN) --platform $(PB_PLATFORM_ARG) --device $(PB_DEVICE_ARG) $(ARGS)
endif

DBG_CMD=$(DEBUGGER) --args $(RUN_CMD)

########################################
# Rules
########################################

default: $(FAILSAFE) $(BUILDDIR) $(BIN) KERNELS

run:
	@echo "Running: $(RUN_CMD)"
	@$(RUN_CMD)

debug: resolvelibOpenCL
	@$(DBG_CMD)

clean :
	rm -f $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi

PB_TIMER_LIB=$(addprefix ${BUILDDIR}/,parboil_opencl.o args.o perfmon.o perf_util.o stub.o) #${BUILDDIR}/parboil_opencl.o ${BUILDDIR}/args.o ${BUILDDIR}/ # $(PARBOIL_ROOT)/common/lib/libparboil_opencl.a

$(BIN) : $(OBJS) ${PB_TIMER_LIB}
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(PARBOIL_ROOT)/common/src/%.c ${BUILDDIR}
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(PARBOIL_ROOT)/common/src/%.cc ${BUILDDIR}
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.c ${BUILDDIR}
	$(CC) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc ${BUILDDIR}
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp ${BUILDDIR}
	$(CXX) $(CXXFLAGS) -c $< -o $@

no_opencl:
	@echo "OPENCL_PATH is not set. Open $(PARBOIL_ROOT)/common/Makefile.conf to set default value."
	@echo "You may use $(PLATFORM_MK) if you want a platform specific configurations."
	@exit 1


