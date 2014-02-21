/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <parboil.h>
#include "file.h"
#include "OpenCL_common.h"

// Possible values are 2, 4, 8 and 16
#define R 2

typedef struct { float a; float b; } float2;
char oclOverhead[] = "OpenCL Overhead";
      
int main( int argc, char **argv ) {

  int n_bytes; 
  int N, B;
  struct pb_TimerSet timers;
  struct pb_Parameters *params;
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {
      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }

  int err = 0;
  if(argc != 3)
    err |= 1;
  else {
    char* numend;
    N = strtol(argv[1], &numend, 10);
    if(numend == argv[1])
      err |= 2;
    B = strtol(argv[2], &numend, 10);
    if(numend == argv[2])
      err |= 4;
  }

  if(err)
  {
    fprintf(stderr, "Expecting two integers for N and B\n");
    exit(-1);
  }

  n_bytes = N*B*sizeof(float2);
    
  pb_InitializeTimerSet(&timers);
  
  pb_AddSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);
  
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  float2 *source    = (float2 *)malloc( n_bytes );
  float2 *result    = (float2 *)calloc( N*B, sizeof(float2) );

  inputData(params->inpFiles[0],(float*)source,N*B*2);

  // OpenCL Code
  cl_int clErrNum;
  pb_Context* pb_context;
  pb_context = pb_InitOpenCLContext(params);
  if (pb_context == NULL) {
    fprintf (stderr, "Error: No OpenCL platform/device can be found."); 
    return -1;
  }

  cl_device_id clDevice = (cl_device_id) pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id) pb_context->clPlatformId;
  cl_context clContext = (cl_context) pb_context->clContext;
  cl_command_queue clCommandQueue;
  
  cl_program clProgram;
  
  cl_kernel fft_kernel;

  cl_mem d_source, d_work;//float2 *d_source, *d_work;
  cl_mem *data0, *data1;

  clCommandQueue = clCreateCommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &clErrNum);
  OCL_ERRCK_VAR(clErrNum);
  
  pb_SetOpenCL(&clContext, &clCommandQueue);
  pb_SwitchToSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);
  
  const char *source_path = "src/opencl_base/fft_kernel.cl";
  char *sourceCode;
  sourceCode = readFile(source_path);
  if (sourceCode == NULL) {
    fprintf(stderr, "Could not load program source of '%s'\n", source_path); exit(1);
  }
  
  clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&sourceCode, NULL, &clErrNum);
  OCL_ERRCK_VAR(clErrNum);
  	  	
  free(sourceCode);
  
  /*
    char compileOptions[1024];
  //                -cl-nv-verbose // Provides register info for NVIDIA devices
  // Set all Macros referenced by kernels
  sprintf(compileOptions, "\
                -D PRESCAN_THREADS=%u\
                -D KB=%u -D UNROLL=%u\
                -D BINS_PER_BLOCK=%u -D BLOCK_X=%u",

                prescanThreads,
                lmemKB, UNROLL,
                bins_per_block, blockX
            ); 
  */
  OCL_ERRCK_RETVAL ( clBuildProgram(clProgram, 1, &clDevice, NULL /*compileOptions*/, NULL, NULL) );
  
  
  char *build_log;
  size_t ret_val_size;
  OCL_ERRCK_RETVAL ( clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size) );
  build_log = (char *)malloc(ret_val_size+1);
  OCL_ERRCK_RETVAL ( clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL) );
  
  // to be careful, terminate with \0
  build_log[ret_val_size] = '\0';

  fprintf(stderr, "%s\n", build_log );
  
  
  fft_kernel = clCreateKernel(clProgram, "GPU_FFT_Global", &clErrNum);
  OCL_ERRCK_VAR(clErrNum);
  
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  // allocate & copy device memory
  d_source = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, n_bytes, source, &clErrNum);  OCL_ERRCK_VAR(clErrNum);
  
  //result is initially zero'd out
  d_work = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, n_bytes, result, &clErrNum);  OCL_ERRCK_VAR(clErrNum);
  


  size_t block[1] = { N/R };
  size_t grid[1] = { B*block[0] };
  
  OCL_ERRCK_RETVAL( clSetKernelArg(fft_kernel, 3, sizeof(int), &N) );

  data0 = &d_source;
  data1 = &d_work;

  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);


  for (int Ns = 1; Ns < N; Ns *= R) {
    OCL_ERRCK_RETVAL( clSetKernelArg(fft_kernel, 0, sizeof(int), &Ns) );
    OCL_ERRCK_RETVAL( clSetKernelArg(fft_kernel, 1, sizeof(cl_mem), (void *)data0) );
    OCL_ERRCK_RETVAL( clSetKernelArg(fft_kernel, 2, sizeof(cl_mem), (void *)data1) ); 

    OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, fft_kernel, 1, 0,
                            grid, block, 0, 0, 0) );
    
    cl_mem *tmp = data0;
    data0 = data1;
    data1 = tmp;
  }
  
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  // copy device memory to host
  //cudaMemcpy(result, d_source, n_bytes,cudaMemcpyDeviceToHost);
  OCL_ERRCK_RETVAL( clEnqueueReadBuffer(clCommandQueue, *data0, CL_TRUE, 
                        0, // Offset in bytes
                        n_bytes, // Size of data to read
                        result, // Host Source
                        0, NULL, NULL) );

  OCL_ERRCK_RETVAL ( clReleaseMemObject(d_source) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(d_work) );

  if (params->outFile) {
    /* Write result to file */
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    outputData(params->outFile, (float*)result, N*B*2);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  }

  free(source);
  free(result);
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  
  pb_DestroyTimerSet(&timers);
  pb_FreeParameters(params);
  
  return 0;
}

