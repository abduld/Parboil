/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <TrioletData.h>

extern "C" {
#include "file.h"
#include "convert_dataset.h"
}

#include "spmv_cxx.h"

using namespace Triolet;

void FromFloatArray(float *h_Ax_vector, Array1<Float> t_Ax_vector, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    h_Ax_vector[i] = t_Ax_vector.at(i);
  }
}

int 
main(int argc, char **argv)
{
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  Triolet_init();



  printf("CPU-based sparse matrix vector multiplication****\n");
  printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
  printf("This version maintained by Chris Rodrigues  ***********\n");
  parameters = pb_ReadParameters(&argc, argv);
  if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL)) {
    fprintf(stderr, "Expecting two input filenames\n");
    exit(-1);
  }
  pb_InitializeTimerSet(&timers);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  //parameters declaration
    int             len;
  int             depth;
  int             dim;
  int             pad = 1;
  int             nzcnt_len;

  //host memory allocation
    // matrix
    float          *h_data;
  int            *h_indices;
  int            *h_ptr;
  int            *h_perm;
  int            *h_nzcnt;
  //vector
    float          *h_Ax_vector;
  float          *h_x_vector;


  //load matrix from files
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
  //inputData(parameters->inpFiles[0], &len, &depth, &dim, &nzcnt_len, &pad,
	      //&h_data, &h_indices, &h_ptr,
	      //&h_perm, &h_nzcnt);
  int             col_count;
  coo_to_jds(
	   parameters->inpFiles[0], //bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
	     1, //row padding
	     pad, //warp size
	     1, //pack size
	     1, //is mirrored ?
	     0, //binary matrix
	     1, //debug level[0 : 2]
	     & h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
	     &col_count, &dim, &len, &nzcnt_len, &depth
    );


  h_Ax_vector = (float *) malloc(sizeof(float) * dim);
  h_x_vector = (float *) malloc(sizeof(float) * dim);
  input_vec(parameters->inpFiles[1], h_x_vector, dim);

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  // Construct Triolet data
  List<Int> t_perm = CreateIntList(dim, h_perm);
  List<Int> t_nzcnt = CreateIntList(nzcnt_len, h_nzcnt);
  List<Int> t_ptr = CreateIntList(dim, h_ptr);
  List<Float> t_data = CreateFloatList(len, h_data);
  List<Int> t_indices = CreateIntList(len, h_indices);
  List<Float> t_x_vector = CreateFloatList(dim, h_x_vector);
  Array1<Float> t_Ax_vector;

  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

  // Run kernel
  int p;
  for (p = 0; p < 50; p++) {
    t_Ax_vector = spmv(t_perm, t_nzcnt, t_ptr, t_data, t_indices, t_x_vector);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  FromFloatArray(h_Ax_vector, t_Ax_vector, dim);

#if 0

  int             p, i, k;
  //main execution
  for (p = 0; p < 50; p++) {
    for (i = 0; i < dim; i++) {
      float           sum = 0.0f;
      //int           bound = h_nzcnt[i / 32];
      int             bound = h_nzcnt[i];
      for (k = 0; k < bound; k++) {
	int             j = h_ptr[k] + i;
	int             in = h_indices[j];

	float           d = h_data[j];
	float           t = h_x_vector[in];

	sum += d * t;
      }
      h_Ax_vector[h_perm[i]] = sum;
    }
  }
#endif

  if (parameters->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    outputData(parameters->outFile, h_Ax_vector, dim);

  }
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(h_data);
  free(h_indices);
  free(h_ptr);
  free(h_perm);
  free(h_nzcnt);
  free(h_Ax_vector);
  free(h_x_vector);
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;

}
