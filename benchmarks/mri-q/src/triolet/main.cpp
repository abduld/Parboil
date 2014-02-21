/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <parboil.h>

#include "TrioletData.h"
extern "C" {
#include "file.h"
}
#include "kernels_cxx.h"
using namespace Triolet;

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

int
main (int argc, char *argv[]) {

  int numX, numK;
  int original_numK;
  float *kx, *ky, *kz;
  float *x, *y, *z;
  float *phiR, *phiI;
  float *Qr, *Qi;

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  pb_InitializeTimerSet(&timers);
  
  Triolet_init(&argc, &argv);

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {
      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }

  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
	{
	  fprintf(stderr, "Expecting an integer parameter\n");
	  exit(-1);
	}

      numK = MIN(inputK, original_numK);
    }

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  {
    /* Create CPU data structures */
    List<Float> pkx = CreateFloatList(numK, kx);
    List<Float> pky = CreateFloatList(numK, ky);
    List<Float> pkz = CreateFloatList(numK, kz);
    List<Float> px = CreateFloatList(numX, x);
    List<Float> py = CreateFloatList(numX, y);
    List<Float> pz = CreateFloatList(numX, z);
    List<Float> pphiR = CreateFloatList(numK, phiR);
    List<Float> pphiI = CreateFloatList(numK, phiI);

    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    List<Tuple<Float,Float,Float,Float> > pphiMag =
      ComputeKValues(pkx,pky,pkz,pphiR, pphiI); 
    List<Tuple<Float, Float> > pQ =
      ComputeQ(pphiMag, px, py, pz);

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    Qr = (float*)malloc(sizeof(float)*numX);
    Qi = (float*)malloc(sizeof(float)*numX);

    {
      int i;
      for (i = 0; i < numX; i++) {
        Qr[i] = pQ.at(i).get<0>();
        Qi[i] = pQ.at(i).get<1>();
      }
    }
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  
  if (params->outFile)
    {
      /* Write Q to file */
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      outputData(params->outFile, Qr, Qi, numX);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

  free(kx);
  free(ky);
  free(kz);
  free(x);
  free(y);
  free(z);
  free(phiR);
  free(phiI);
  free(Qr);
  free(Qi);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);

  return 0;
}
