# (c) 2010 The Board of Trustees of the University of Illinois.

LANGUAGE=c
SRCDIR_OBJS=main.o 

# To use slow (but precise) math routines
ifeq ($(slowmath), 1)
	APP_CFLAGS += -fp-model precise
else
	APP_CFLAGS += -ffast-math
	APP_CXXFLAGS += -ffast-math
endif

APP_CFLAGS += -fopenmp -O3
APP_CXXFLAGS += -fopenmp -O3

APP_LDFLAGS = -lm -lstdc++



