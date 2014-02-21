/***************************************************************************
 *cr
 *cr            (C) Copyright 2012-2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <iostream>
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <parboil.h>

#include "file.h"

#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     printf("Error: %s!\n",errorMessage);   \
     printf("Line: %d\n",__LINE__);         \
     exit(1);                               \
  }

char* readFile(const char* fileName)
{
        FILE* fp;
        fp = fopen(fileName,"r");
        if(fp == NULL)
        {
                printf("Error: Cannot open kernel file for reading!\n");
                exit(1);
        }
        fseek(fp,0,SEEK_END);
        long size = ftell(fp);
        rewind(fp);
        char* buffer = (char*)malloc(sizeof(char)*(size+1));
        if(buffer  == NULL)
        {
                printf("Error: Cannot allocated buffer for file contents!\n");
                fclose(fp);
                exit(1);
        }
        size_t res = fread(buffer,1,size,fp);
        if(res != size)
        {
                printf("Error: Cannot read kernel file contents!\n");
                fclose(fp);
                exit(1);
        }
	buffer[size] = 0;
        fclose(fp);
        return buffer;
}

void
histo_R_per_block(cl_mem histo, // Output histogram on device
                  cl_mem data,  // Input data on device
                  int size,            // Input data size
                  int NUM_BLOCKS,      // Number of GPU thread blocks
                  int THREADS,         // Number of GPU threads per block
                  int BINS,   // Number of histogram bins to use
                  struct pb_TimerSet *timers, 
                  cl_command_queue clCommandQueue, cl_kernel clKernel)
{
  size_t dimGrid = NUM_BLOCKS*THREADS;
  size_t dimBlock = THREADS;
  int shmem_bytes = BINS * sizeof(int);

  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  /* Parameters */
  cl_int clStatus;
  clStatus = clSetKernelArg(clKernel,0,sizeof(cl_mem),&histo);
  clStatus = clSetKernelArg(clKernel,1,sizeof(cl_mem),&data);
  clStatus = clSetKernelArg(clKernel,2,shmem_bytes,0);
  clStatus = clSetKernelArg(clKernel,3,sizeof(int),&size);
  clStatus = clSetKernelArg(clKernel,4,sizeof(int),&BINS);
  CHECK_ERROR("clSetKernelArg")

  /* Launch kernel */
  clStatus = clEnqueueNDRangeKernel(clCommandQueue,clKernel,1,NULL,&dimGrid,&dimBlock,0,NULL,NULL);
  CHECK_ERROR("clEnqueueNDRangeKernel")

  /* Finish */
  clStatus = clFinish(clCommandQueue);
  CHECK_ERROR("clFinish")

}


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

  /* Context, create kernel */
  cl_int clStatus;
  pb_Context* pb_context;
  pb_context = pb_InitOpenCLContext(parameters);
  if (pb_context == NULL) {
    fprintf (stderr, "Error: No OpenCL platform/device can be found."); 
    return -1;
  }

  cl_device_id clDevice = (cl_device_id) pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id) pb_context->clPlatformId;
  cl_context clContext = (cl_context) pb_context->clContext;

  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  CHECK_ERROR("clCreateCommandQueue")

  pb_SetOpenCL(&clContext, &clCommandQueue);

  const char* clSource[] = {readFile("src/opencl_base/kernel.cl")};
  cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  char clOptions[50];
  sprintf(clOptions,"-I src/opencl_base");

  clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
  if (clStatus != CL_SUCCESS) {
    size_t string_size = 0;
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 
                          0, NULL, &string_size);
    char* string = (char *) malloc(string_size*sizeof(char));
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 
                          string_size, string, NULL);
    std::cerr << string;
  }
  CHECK_ERROR("clBuildProgram")

  cl_kernel clKernel = clCreateKernel(clProgram,"histo_R_per_block_kernel",&clStatus);
  CHECK_ERROR("clCreateKernel")
  
  /* Variable declarations */
  unsigned int *h_DataA;
  unsigned int *h_ResultGPU_block;

  const int DATA_W = atoi(argv[2]);
  const int DATA_H = atoi(argv[3]);
  int DATA_SIZE = DATA_W * DATA_H;
  const int DATA_SIZE_INT = DATA_SIZE * sizeof(unsigned int);

  int BINS = atoi(argv[4]);

  cl_uint compute_units;
  clGetDeviceInfo(clDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
  int NUM_BLOCKS = 10*compute_units; // for 10% load imbalance
  size_t THREADS;
  clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(THREADS), &THREADS, NULL);
  int repeat = atoi(argv[5]);

  /* Print input information */
  char device_string[1024];
  clGetDeviceInfo(clDevice, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("-------------------------------------------------------------------------------\n");
  printf("Running %s on %s\n", argv[0], device_string);
  printf("%d repetitions\n", repeat);
  printf("Image size = %d pixels (%dx%d)\n", DATA_SIZE, DATA_W, DATA_H);
  printf("Histogram bins: %d\n", BINS);
  printf("Execution configuration: %d blocks of %d threads, 1 sub-histogram per block\n", NUM_BLOCKS, THREADS);
  printf("-------------------------------------------------------------------------------\n");

  /* Allocate host memory */
  h_DataA = (unsigned int*) malloc(DATA_SIZE_INT);
  h_ResultGPU_block = (unsigned int*) malloc(BINS*sizeof(unsigned int));
  memset(h_ResultGPU_block, 0, BINS*sizeof(unsigned int));

  /* Allocate device memory */
  cl_mem d_DataA;
  d_DataA = clCreateBuffer(clContext,CL_MEM_READ_ONLY,DATA_SIZE_INT,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  cl_mem d_histo_block;
  d_histo_block = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY,BINS*sizeof(unsigned int),NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")

  /* Initizalize host array */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  readImage(parameters->inpFiles[0], h_DataA, DATA_SIZE);

  /* Copy host to device */
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  clStatus = clEnqueueWriteBuffer(clCommandQueue,d_DataA,CL_TRUE,0,DATA_SIZE_INT,h_DataA,0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")


  for(int i = 0; i < repeat; i++) {
    /* Clear output */
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    clStatus = clEnqueueWriteBuffer(clCommandQueue,d_histo_block,CL_TRUE,0,BINS*sizeof(unsigned int),h_ResultGPU_block,0,NULL,NULL);
    CHECK_ERROR("clEnqueueWriteBuffer")
	
    /* Run histogram on GPU */
    histo_R_per_block(d_histo_block, d_DataA, DATA_SIZE, NUM_BLOCKS, THREADS, BINS,
                      &timers,clCommandQueue,clKernel);
  }


  if(parameters->outFile) {
    /* Copy the result back to the host */
    pb_SwitchToTimer( &timers, pb_TimerID_COPY );
    clStatus = clEnqueueReadBuffer(clCommandQueue,d_histo_block,CL_TRUE,0,BINS*sizeof(int),h_ResultGPU_block,0,NULL,NULL);
    CHECK_ERROR("clEnqueueReadBuffer")

    /* Write the result */
    pb_SwitchToTimer(&timers,  pb_TimerID_IO);
    writeVector(parameters->outFile, h_ResultGPU_block, BINS);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  /* Free device memory */
  clStatus = clReleaseMemObject(d_DataA);
  clStatus = clReleaseMemObject(d_histo_block);
  clStatus = clReleaseKernel(clKernel);
  clStatus = clReleaseProgram(clProgram);
  clStatus = clReleaseCommandQueue(clCommandQueue);
  clStatus = clReleaseContext(clContext);
  CHECK_ERROR("clReleaseContext")

  free((void*)clSource[0]);

  /* Free host memory */
  free(h_DataA);
  free(h_ResultGPU_block);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;

}

