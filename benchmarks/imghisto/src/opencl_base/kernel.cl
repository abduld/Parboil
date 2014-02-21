/***************************************************************************
 *cr
 *cr            (C) Copyright 2012-2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

///////////////////////////////////////////////////////////////////////////////
// R-per-block approach.- Base version: 1 sub-histogram per block
//
// histo:	Final histogram in global memory
// data:	Input image. Pixels are stored in 4-byte unsigned int
// size:	Input image size (number of pixels)
// BINS:	Number of histogram bins
//
// This function was developed at the University of Córdoba and
// contributed by Juan Gómez-Luna.
///////////////////////////////////////////////////////////////////////////////
__kernel void histo_R_per_block_kernel(__global unsigned int* histo,
                                         __global unsigned int* data,
                                         __local unsigned int* Hs,
                                         int size, int BINS)
{
  // Work-group and work-item index
  const int bx = get_group_id(0);
  const int tx = get_local_id(0);

  // Constants for naive read access
  const int begin = bx * get_local_size(0) + tx;
  const int end = size;
  const int step = get_global_size(0);

  // Sub-histogram initialization
  for(int pos = tx; pos < BINS; pos += get_local_size(0)) Hs[pos] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);	// Intra-group synchronization

  // Main loop
  for(int i = begin; i < end; i += step){
    // Global memory read
    unsigned int d = data[i];

    // Atomic vote in local memory
    atom_inc(&Hs[(d * BINS) >> 12]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);	// Intra-group synchronization

  // Merge in global memory
  for(int pos = tx; pos < BINS; pos += get_local_size(0)){
    unsigned int sum = 0;
    sum = Hs[pos];
    // Atomic addition in global memory
    atom_add(histo + pos, sum);
  }
}

