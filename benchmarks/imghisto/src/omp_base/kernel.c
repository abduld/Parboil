/***************************************************************************
 *cr
 *cr            (C) Copyright 2012-2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <math.h>
#include <stdio.h>
#include <omp.h>

///////////////////////////////////////////////////////////////////////////////
// Histogram CPU
///////////////////////////////////////////////////////////////////////////////
void histo_CPU(unsigned int *histo, unsigned int *data, int size, int BINS) {

  int num_threads = omp_get_max_threads();
#pragma omp parallel
{
  int i;
  unsigned int private_histo[BINS];
  
  //initialize the private histo
  for(i=0;i<BINS;i++)
    private_histo[i] = 0;
#pragma omp barrier

//accumulate values into private histos
#pragma omp for
  for(i = 0; i < size; i++) {
    private_histo[(data[i] * BINS) >> 12]++;
  }

  //combine the results
  for(i=0;i<BINS;i++) {
#pragma omp atomic
    histo[i] += private_histo[i];
  }
}
}

