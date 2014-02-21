/***************************************************************************
 *cr
 *cr            (C) Copyright 2012-2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <math.h>
#include <stdio.h>

///////////////////////////////////////////////////////////////////////////////
// Histogram CPU
///////////////////////////////////////////////////////////////////////////////
void histo_CPU(unsigned int *histo, unsigned int *data, int size, int BINS) {
  int i;
  for(i = 0; i < size; i++){
    // Read pixel
    unsigned int d = ((data[i] * BINS) >> 12);

    // Vote in histogram
    histo[d]++;
  }
}


