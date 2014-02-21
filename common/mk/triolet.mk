# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include
LANG_CXXFLAGS=$(LANG_CFLAGS)
LANG_LDFLAGS=-L$(PARBOIL_ROOT)/common/lib -lparboil

CFLAGS=$(OPT_CFLAGS) $(LANG_CFLAGS) $(PLATFORM_CFLAGS) $(APP_CFLAGS)
CXXFLAGS=$(OPT_CFLAGS) $(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS) $(APP_CXXFLAGS) -I$(BUILDDIR)
TRIOLETFLAGS=$(LANG_TRIOLETFLAGS) $(PLATFORM_TRIOLETFLAGS) $(APP_TRIOLETFLAGS)
LDFLAGS=$(OPT_LDFLAGS) $(LANG_LDFLAGS) $(PLATFORM_LDFLAGS) $(APP_LDFLAGS)

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

ifeq ($(TRIOLETHOME),)
FAILSAFE=no_triolet
else 
FAILSAFE=
endif

########################################
# Derived variables
########################################

TRIOLET_OBJS=$(call INBUILDDIR, $(addsuffix .o,$(TRIOLET_SRCS)))

TRIOLET_APP_HEADERS=$(call INBUILDDIR, $(addsuffix _cxx.h,$(TRIOLET_SRCS)))

OBJS = $(TRIOLET_OBJS) $(call INBUILDDIR,$(SRCDIR_OBJS))

ifeq ($(DEBUGGER),)
DEBUGGER=gdb
endif

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
	@echo $(TRIOLET_APP_HEADERS)
	@$(shell echo $(RUNTIME_ENV)) ldd $(BIN)
	@$(shell echo $(RUNTIME_ENV)) $(BIN) $(ARGS)

debug:
	@$(shell echo $(RUNTIME_ENV)) ldd $(BIN)
	@$(shell echo $(RUNTIME_ENV)) $(DEBUGGER) --args $(BIN) $(ARGS)

clean :
	rm -f $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi

$(BIN) : $(OBJS) $(PARBOIL_ROOT)/common/lib/libparboil.a
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c $(TRIOLET_APP_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.tri
	$(TRIOLETC) $(TRIOLETFLAGS) -o $@ $<

# This rule is for dependences only
$(BUILDDIR)/%_cxx.h : $(BUILDDIR)/%.o $(SRCDIR)/%.tri
	@true

no_triolet:
	@echo "TRIOLETHOME is not set. Open $(PARBOIL_ROOT)/common/Makefile.conf to set default value."
	@echo "You may use $(PLATFORM_MK) if you want a platform specific configurations."
	@exit 1

