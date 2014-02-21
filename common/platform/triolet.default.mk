# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using gcc for C++
# code and the triolet compiler for triolet code

# gcc (default)
CC = gcc
PLATFORM_CFLAGS = -I$(TRIOLETHOME)/share/triolet-0.1/include -I/usr/include/gc

CXX = g++
PLATFORM_CXXFLAGS = -I$(TRIOLETHOME)/share/triolet-0.1/include -I/usr/include/gc

TRIOLETC = $(TRIOLETHOME)/bin/triolet
PLATFORM_TRIOLETFLAGS =

LINKER = g++
PLATFORM_LDFLAGS = -L$(TRIOLETHOME)/share/triolet-0.1 -ltrioletrts -lgc -lpthread -lm

