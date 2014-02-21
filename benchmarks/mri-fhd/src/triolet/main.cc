/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * CUDA code for creating the FHD data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 * 
 * recommended g++ options:
 *   -O3 -lm -ffast-math -funroll-all-loops
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *     function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/time.h>

#include <parboil.h>

#include "TrioletData.h"

#include "file.h"

#include "kernels_cxx.h"
using namespace Triolet;

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

typedef List<Tuple<Float, Float, Float, Tuple<Float, Float> > > KValues;

int
main (int argc, char *argv[])
{
  int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *dR, *dI;		/* D values (complex) */
  float *Outr, *Outi;		/* Output signal (complex) */

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  Triolet_init();

  pb_InitializeTimerSet(&timers);

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
	    &phiR, &phiI,
	    &dR, &dI);

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
    List<Float> pdR = CreateFloatList(numK, dR);
    List<Float> pdI = CreateFloatList(numK, dI);

    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    KValues kvalues =
      ComputeKValues(pkx, pky, pkz, pphiR, pdR, pphiI, pdI);

    List<Tuple<Float, Float> > pOut =
        ComputeFH(kvalues, px, py, pz);

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    Outr = (float*)malloc(sizeof(float)*numX);
    Outi = (float*)malloc(sizeof(float)*numX);

    {
      int i;
      for (i = 0; i < numX; i++) {
        Outr[i] = pOut.at(i).get<0>();
        Outi[i] = pOut.at(i).get<1>();
      }
    }
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  if (params->outFile)
    {
      /* Write result to file */
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      outputData(params->outFile, Outr, Outi, numX);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (dR);
  free (dI);
  free (Outr);
  free (Outi);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);

  return 0;
}
