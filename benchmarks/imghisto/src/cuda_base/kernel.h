/***************************************************************************
 *cr
 *cr            (C) Copyright 2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <parboil.h>

void
histo_R_per_block(unsigned int* histo,
                  unsigned int* data,
                  int size,
                  int NUM_BLOCKS,
                  int THREADS,
                  int BINS,
                  struct pb_TimerSet *timers);
