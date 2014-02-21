# (c) 2007 The Board of Trustees of the University of Illinois.

LANGUAGE=cuda
SRCDIR_OBJS=main.o file.o

APP_LDFLAGS = -lm -lstdc++

# To use slow (but precise) math routines
ifeq ($(slowmath), 1)
	APP_CFLAGS = -fp-model precise
else
	APP_CFLAGS = -ffast-math
	APP_CXXFLAGS = -ffast-math
	APP_CUDACFLAGS = --use_fast_math
endif

# To use the MKL library
ifeq ($(mkl), 1)
	APP_LDFLAGS += -L/opt/intel/mkl/9.1.023/lib/32 \
		-lmkl -lvml -lguide -lpthread
	APP_CFLAGS = -DMKL -I/opt/intel/mkl/9.1.023/include
endif


