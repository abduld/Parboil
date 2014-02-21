# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include
LANG_CXXFLAGS=$(LANG_CFLAGS)
LANG_LDFLAGS=

CFLAGS=$(OPT_CFLAGS) $(LANG_CFLAGS) $(PLATFORM_CFLAGS) $(APP_CFLAGS)
CXXFLAGS=$(OPT_CFLAGS) $(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS) $(APP_CXXFLAGS)
LDFLAGS=$(OPT_CFLAGS) $(LANG_LDFLAGS) $(PLATFORM_LDFLAGS) $(APP_LDFLAGS)

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

default: $(BUILDDIR) $(BIN)

run:
	mpirun -np `cat /proc/cpuinfo | grep processor | wc -l` $(BIN) $(ARGS)

debug:
	$(DEBUGGER) --args mpirun -np `cat /proc/cpuinfo | grep processor | wc -l` $(BIN) $(ARGS)

clean :
	rm -f $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi

$(BIN) : $(OBJS) $(BUILDDIR)/parboil.o $(BUILDDIR)/args.o
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/parboil.o: $(PARBOIL_ROOT)/common/src/parboil.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/args.o : $(PARBOIL_ROOT)/common/src/args.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

