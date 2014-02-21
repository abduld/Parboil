/***************************************************************************
 *cr
 *cr            (C) Copyright 2012-2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <parboil.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "file.h"
#include "kernel.c"

int main(int argc, char **argv) {

  /* Parboil parameters */
  struct pb_Parameters *parameters;
  parameters = pb_ReadParameters(&argc, argv);

  /* Parboil timing */
  struct pb_TimerSet timers;
  pb_InitializeTimerSet(&timers);

  /* Syntax verification */
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  if (argc != 6) {
    printf("Wrong input parameters\n");
    printf("Parameters: -- DEVICE DATA_W DATA_H BINS REPITITIONS\n");
    exit(1);
  }

  /* Variable declarations */
  unsigned int *h_DataA; 
  unsigned int *h_ResultGPU_block;

  const int DATA_W = atoi(argv[2]);
  const int DATA_H = atoi(argv[3]);
  int DATA_SIZE = DATA_W * DATA_H;
  const int DATA_SIZE_INT = DATA_SIZE * sizeof(unsigned int);
  int BINS = atoi(argv[4]);
  int repeat = atoi(argv[5]);

  /* Print input information */
  printf("-------------------------------------------------------------------------------\n");
  printf("%d repetitions\n", repeat);
  printf("Image size = %d pixels (%dx%d)\n", DATA_SIZE, DATA_W, DATA_H);
  printf("Histogram bins: %d\n", BINS);
  printf("-------------------------------------------------------------------------------\n");

  /* Allocate host memory */
  h_DataA = (unsigned int*) malloc(DATA_SIZE_INT);
  h_ResultGPU_block = (unsigned int*) malloc(BINS*sizeof(unsigned int));

  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  readImage(parameters->inpFiles[0], h_DataA, DATA_SIZE);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  int i;
  for(i = 0; i < repeat; i++){

    memset(h_ResultGPU_block,0,BINS*sizeof(unsigned int));
    
    /* Launch kernel */
    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    histo_CPU(h_ResultGPU_block, h_DataA, DATA_SIZE, BINS);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  }

  if(parameters->outFile) {

    /* Write the result */
    pb_SwitchToTimer(&timers,  pb_TimerID_IO);
    writeVector(parameters->outFile, h_ResultGPU_block, BINS);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  }


  /* Free host memory */
  free(h_DataA);
  free(h_ResultGPU_block);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;

}

