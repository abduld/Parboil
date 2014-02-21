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

#include "file.h"
#include "kernel.h"

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

  /* Check the compute capability of the device */
  int num_devices=0;
  cudaGetDeviceCount(&num_devices);
  if(0 == num_devices){
    printf("Your system does not have a CUDA capable device\n");
    exit(1);
  }
  int device = atoi(argv[1]);
  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties,device);
  if((1 == device_properties.major)&&(device_properties.minor < 2)) {
    printf("%s does not have compute capability 1.2 or later\n\n",device_properties.name);
  }

  /* Variable declarations */
  unsigned int *h_DataA; 
  unsigned int *d_DataA; 
  unsigned int *d_Temp;
  unsigned int *d_histo_block;
  unsigned int *h_ResultGPU_block;

  const int DATA_W = atoi(argv[2]);
  const int DATA_H = atoi(argv[3]);
  int DATA_SIZE = DATA_W * DATA_H;
  const int DATA_SIZE_INT = DATA_SIZE * sizeof(unsigned int);

  int BINS = atoi(argv[4]);

  int NUM_BLOCKS = 10*device_properties.multiProcessorCount; // for 10% load imbalance
  int THREADS = device_properties.maxThreadsPerBlock;
  int repeat = atoi(argv[5]);

  /* Print input information */
  printf("-------------------------------------------------------------------------------\n");
  printf("Running %s on %s\n", argv[0], device_properties.name);
  printf("%d repetitions\n", repeat);
  printf("Image size = %d pixels (%dx%d)\n", DATA_SIZE, DATA_W, DATA_H);
  printf("Histogram bins: %d\n", BINS);
  printf("Execution configuration: %d blocks of %d threads, 1 sub-histogram per block\n", NUM_BLOCKS, THREADS);
  printf("-------------------------------------------------------------------------------\n");

  /* Allocate host memory */
  cudaMallocHost((void **)&h_DataA, DATA_SIZE_INT);
  cudaMallocHost((void **)&h_ResultGPU_block, BINS*sizeof(unsigned int));

  /* Allocate device memory */
  cudaMalloc((void **)&d_DataA, DATA_SIZE_INT);
  cudaMalloc((void **)&d_Temp, BINS*sizeof(unsigned int));
  cudaMalloc((void **)&d_histo_block, BINS*sizeof(unsigned int));

  /* Initizalize host array */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  readImage(parameters->inpFiles[0], h_DataA, DATA_SIZE);

  /* Copy host to device */
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  cudaMemcpy(d_DataA, h_DataA, DATA_SIZE_INT, cudaMemcpyHostToDevice);

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  for(int i = 0; i < repeat; i++) {
    /* Run histogram on GPU */
    histo_R_per_block(d_histo_block, d_DataA, DATA_SIZE, NUM_BLOCKS, THREADS,
                      BINS, &timers);
  }

  if(parameters->outFile) {

    /* Copy the result back to the host */
    cudaMemcpy(h_ResultGPU_block,d_histo_block,BINS*sizeof(int),cudaMemcpyDeviceToHost);

    /* Write the result */
    pb_SwitchToTimer(&timers,  pb_TimerID_IO);
    writeVector(parameters->outFile, h_ResultGPU_block, BINS);

  }


  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  /* Free device memory */
  cudaFree(d_Temp);
  cudaFree(d_DataA);
  cudaFree(d_histo_block);

  /* Free host memory */
  cudaFreeHost(h_DataA);
  cudaFreeHost(h_ResultGPU_block);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;

}

