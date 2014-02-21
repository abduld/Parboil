/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifdef __MCUDA__
#include <mcuda.h>
#else
#include <cuda.h>
#endif

/* Size of a thread block */
#define BLOCKSIZEX 16
#define BLOCKSIZEY 8

/* Number of grid points processed by a thread */
#define UNROLLX 8

/* Number of atoms processed by a kernel */
#define MAXATOMS 4000

/* Size of the benchmark problem.  The GPU can run larger problems in a
 * reasonable time.
 *
 * For VOLSIZEX, VOLSIZEY, size 1024 is suitable for a few seconds of
 * GPU computation and size 128 is suitable for a few seconds of
 * CPU computation.
 *
 * For ATOMCOUNT, 100000 is suitable for GPU computation and 10000 is
 * suitable for CPU computation.
 */
#define VOLSIZEX 512
#define VOLSIZEY 512
#define ATOMCOUNT 40000

#ifdef __MCUDA__

#define CUERR

#else

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}

#endif

__global__ void cenergy(int numatoms,
			float gridspacing,
			float * energygrid);

int copyatomstoconstbuf(float *atoms,
			int count,
			float zplane);
