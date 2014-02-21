
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
#include <string.h>
#include <TrioletData.h>

#include "file.h"

extern "C" {
#include "common.h"
}

#include "stencil_cxx.h"

using namespace Triolet;

static Array3<Float>
read_data(int nx,int ny,int nz,FILE *fp) 
{
  Incomplete<Array3<Float> > make_arr;
  make_arr.create(0, nz, 0, ny, 0, nx);
  int i, j, k;
  float tmp;
  for(i=0;i<nz;i++) {
    for(j=0;j<ny;j++) {
      for(k=0;k<nx;k++) {
        fread(&tmp, sizeof(float), 1, fp);
        make_arr.at(i,j,k) = tmp;
      }
    }
  }
  return make_arr.freeze();
}

int main(int argc, char** argv) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;

        Triolet_init();
	
	printf("CPU-based 7 points stencil codes****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and I-Jui Sung<sung10@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);

	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//declaration
	int nx,ny,nz;
	int size;
    int iteration;
	float c0=1.0f/6.0f;
	float c1=1.0f/6.0f/6.0f;

	if (argc<5) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
		  "t: the iteration time\n");
      return -1;
    }

	nx = atoi(argv[1]);
	if (nx<1)
		return -1;
	ny = atoi(argv[2]);
	if (ny<1)
		return -1;
	nz = atoi(argv[3]);
	if (nz<1)
		return -1;
	iteration = atoi(argv[4]);
	if(iteration<1)
		return -1;

	
	size=nx*ny*nz;
	
  FILE *fp = fopen(parameters->inpFiles[0], "rb");
  Array3<Float> edges = read_data(nx,ny,nz,fp);
  fclose(fp);

  // Get a non-edge copy of the array
  Array3<Float> middle = get_inner_array(edges);

  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
  int t;
  for(t=0;t<iteration;t++) {
    middle = stencil(c0, c1, edges, middle);
  }
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  Array3<Float> A = combine_arrays(edges, middle);
 
  if (parameters->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    outputData(parameters->outFile,A,nx,ny,nz);
  }
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	return 0;

}
