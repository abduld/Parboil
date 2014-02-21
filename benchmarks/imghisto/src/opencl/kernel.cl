/***************************************************************************
 *cr
 *cr            (C) Copyright 2012-2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

// Constants
#define WARP_SIZE 32

///////////////////////////////////////////////////////////////////////////////
// R-per-block approach
// Replication + Padding + Interleaved read access
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
                                         int size, int BINS, int BINSp, int R)
{
  // Work-group and work-item index
  const int bx = get_group_id(0);
  const int tx = get_local_id(0);
  // Warp and lane
  const unsigned int warpid = tx >> 5;
  const unsigned int lane = tx & 31;	

  // Offset to per-block sub-histograms
  const unsigned int off_rep = BINSp * (tx % R);

  // Constants for interleaved read access
  const int warps_block = get_local_size(0) / WARP_SIZE;
  const int begin = (size / warps_block) * warpid + WARP_SIZE * bx + lane;
  const int end = (size / warps_block) * (warpid + 1);
  const int step = WARP_SIZE * (get_global_size(0) / get_local_size(0));
  // Constants for naive read access
  /*const int begin = bx * get_local_size(0) + tx;
  const int end = size;
  const int step = get_global_size(0);*/

  // Sub-histogram initialization
  for(int pos = tx; pos < BINSp*R; pos += get_local_size(0)) Hs[pos] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);	// Intra-group synchronization

  // Main loop
  for(int i = begin; i < end; i += step){
    // Global memory read
    unsigned int d = data[i];

    // Atomic vote in local memory
    atom_inc(&Hs[off_rep + ((d * BINS) >> 12)]);

  }

  barrier(CLK_LOCAL_MEM_FENCE);	// Intra-group synchronization

  // Merge in global memory
  for(int pos = tx; pos < BINS; pos += get_local_size(0)){
    unsigned int sum = 0;
    for(int base = 0; base < BINSp*R; base += BINSp)
      sum += Hs[base + pos];
    // Atomic addition in global memory
    atom_add(histo + pos, sum);
  }
}

