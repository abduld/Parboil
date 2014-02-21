/***************************************************************************
 *cr
 *cr            (C) Copyright 2012-2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <cuda.h>
#include <kernel.h>

__global__ void histo_R_per_block_kernel(unsigned int* histo,
                                         unsigned int* data,
                                         int size, int BINS);

void
histo_R_per_block(unsigned int* histo, // Output histogram on device
                  unsigned int* data,  // Input data on device
                  int size,            // Input data size
                  int NUM_BLOCKS,      // Number of GPU thread blocks
                  int THREADS,         // Number of GPU threads per block
                  int BINS,   // Number of histogram bins to use
                  struct pb_TimerSet *timers)
{
  /* Clear output */
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  cudaMemset(histo, 0, BINS*sizeof(unsigned int));

  /* Launch kernel */
  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  int shmem_bytes = BINS * sizeof(int);
  histo_R_per_block_kernel<<<dim3(NUM_BLOCKS), dim3(THREADS), shmem_bytes>>>
    (histo, data, size, BINS);

  /* Check for errors */
  cudaError_t err = cudaGetLastError();
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  /* Synchronize so that subsequent timer measurements are accurate */
  cudaDeviceSynchronize();
}

// Constants
#define WARP_SIZE 32

// Dynamic shared memory allocation
extern __shared__ unsigned int Hs[];

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
__global__ void histo_R_per_block_kernel(unsigned int* histo,
                                         unsigned int* data,
                                         int size, int BINS)
{
  // Block and thread index
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;

  // Constants for naive read access
  const int begin = bx * blockDim.x + tx;
  const int end = size;
  const int step = blockDim.x * gridDim.x;

  // Sub-histogram initialization
  for(int pos = tx; pos < BINS; pos += blockDim.x) Hs[pos] = 0;

  __syncthreads();	// Intra-block synchronization

  // Main loop
  for(int i = begin; i < end; i += step){
    // Global memory read
    unsigned int d = data[i];

    // Atomic vote in shared memory
    atomicAdd(&Hs[(d * BINS) >> 12], 1);
  }

  __syncthreads();	// Intra-block synchronization

  // Merge in global memory
  for(int pos = tx; pos < BINS; pos += blockDim.x){
    unsigned int sum = 0;
    sum = Hs[pos];
    // Atomic addition in global memory
    atomicAdd(histo + pos, sum);
  }
}


