/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#include "util.h"    
#include "OpenCL_common.h"

#define DEFAULT_BLOCK_X         14
#define DEFAULT_FINAL_THREADS 512

/******************************************************************************
* Implementation: GPU
* Details:
* in the GPU implementation of histogram, we begin by computing the span of the
* input values into the histogram. Then the histogramming computation is carried
* out by a (BLOCK_X, BLOCK_Y) sized grid, where every group of Y (same X)
* computes its own partial histogram for a part of the input, and every Y in the
* group exclusively writes to a portion of the span computed in the beginning.
* Finally, a reduction is performed to combine all the partial histograms into
* the final result.
******************************************************************************/

int main(int argc, char* argv[]) {
  struct pb_Parameters *parameters;

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if(!parameters->inpFiles[0]){
    fputs("Input file expected\n", stderr);
    return -1;
  }

  
  struct pb_TimerSet timers;
  
  char oclOverhead[] = "OCL Overhead";
  char intermediates[] = "IntermediatesKernel";
  char finals[] = "FinalKernel";

  pb_InitializeTimerSet(&timers);
  
  pb_AddSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, intermediates, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, finals, pb_TimerID_KERNEL);
    
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  
  int numIterations;
  if (argc >= 2){
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    return -1;
  }

  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;

  FILE* f = fopen(parameters->inpFiles[0],"rb");
  int result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }

  unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
  unsigned char* histo = (unsigned char*) calloc (histo_width*histo_height, sizeof(unsigned char));

  result = fread(img, sizeof(unsigned int), img_width*img_height, f);

  fclose(f);

  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }

  cl_int ciErrNum;
  pb_Context* pb_context;
  pb_context = pb_InitOpenCLContext();
  if (pb_context == NULL) {
    fprintf (stderr, "Error: No OpenCL platform/device can be found."); 
    return -1;
  }

  cl_int clStatus;
  cl_device_id clDevice = (cl_device_id) pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id) pb_context->clPlatformId;
  cl_context clContext = (cl_context) pb_context->clContext;
  cl_command_queue clCommandQueue;
  
  cl_program clProgram[2];
  
  cl_kernel histo_intermediates_kernel;
  cl_kernel histo_final_kernel;
  
  cl_mem input;
  cl_mem ranges;
  cl_mem sm_mappings;
  cl_mem global_subhisto;
  cl_mem global_overflow;
  cl_mem final_histo;
  
  clCommandQueue = clCreateCommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  
  pb_SetOpenCL(&clContext, &clCommandQueue);
  pb_SwitchToSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);

  cl_uint workItemDimensions;
  OCL_ERRCK_RETVAL( clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &workItemDimensions, NULL) );
  
  size_t workItemSizes[workItemDimensions];
  OCL_ERRCK_RETVAL( clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES, workItemDimensions*sizeof(size_t), workItemSizes, NULL) );
  
  size_t program_length[2];
  const char *source_path[2] = { 
    "src/opencl_naive/histo_intermediates.cl", 
   "src/opencl_naive/histo_final.cl"};
  char *source[4];

  for (int i = 0; i < 2; ++i) {
    // Dynamically allocate buffer for source
    source[i] = oclLoadProgSource(source_path[i], "", &program_length[i]);
    if(!source[i]) {
      fprintf(stderr, "Could not load program source\n"); exit(1);
    }
  	
  	clProgram[i] = clCreateProgramWithSource(clContext, 1, (const char **)&source[i], &program_length[i], &ciErrNum);
  	OCL_ERRCK_VAR(ciErrNum);
  	  	
  	free(source[i]);
  }
  	
  	  	  	  	  	  	  	
  for (int i = 0; i < 2; ++i) {
    //fprintf(stderr, "Building Program #%d...\n", i);
    OCL_ERRCK_RETVAL ( clBuildProgram(clProgram[i], 1, &clDevice, NULL, NULL, NULL) );
       
    #if 1
       char *build_log;
       size_t ret_val_size;
       ciErrNum = clGetProgramBuildInfo(clProgram[i], clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);	OCL_ERRCK_VAR(ciErrNum);
       build_log = (char *)malloc(ret_val_size+1);
       ciErrNum = clGetProgramBuildInfo(clProgram[i], clDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
       	OCL_ERRCK_VAR(ciErrNum);
       	

       // to be carefully, terminate with \0
       // there's no information in the reference whether the string is 0 terminated or not
       build_log[ret_val_size] = '\0';

       fprintf(stderr, "%s\n", build_log );
     #endif
  }
  	
  histo_intermediates_kernel = clCreateKernel(clProgram[0], "histo_intermediates_kernel", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  histo_final_kernel = clCreateKernel(clProgram[1], "histo_final_kernel", &ciErrNum);
  OCL_ERRCK_VAR(ciErrNum);
  
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);  

  input =           clCreateBuffer(clContext, CL_MEM_READ_WRITE, 
      img_width*img_height*sizeof(unsigned int), NULL, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);
  ranges =          clCreateBuffer(clContext, CL_MEM_READ_WRITE, 2*sizeof(unsigned int), NULL, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);  
  sm_mappings =     clCreateBuffer(clContext, CL_MEM_READ_WRITE, img_width*img_height*4*sizeof(unsigned char), NULL, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);
  global_subhisto = clCreateBuffer(clContext, CL_MEM_READ_WRITE, histo_width*histo_height*sizeof(unsigned int), NULL, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);
  global_overflow = clCreateBuffer(clContext, CL_MEM_READ_WRITE, histo_width*histo_height*sizeof(unsigned int), NULL, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);
  final_histo =     clCreateBuffer(clContext, CL_MEM_READ_WRITE, histo_width*histo_height*sizeof(unsigned char), NULL, &ciErrNum); OCL_ERRCK_VAR(ciErrNum);

  // Must dynamically allocate. Too large for stack
  unsigned int *zeroData;
  zeroData = (unsigned int *) calloc(img_width*histo_height, sizeof(unsigned int));
  if (zeroData == NULL) {
    fprintf(stderr, "Failed to allocate %ld bytes of memory on host!\n", sizeof(unsigned int) * img_width * histo_height);
    exit(1);
  }
   
  for (int y=0; y < img_height; y++){
    OCL_ERRCK_RETVAL( clEnqueueWriteBuffer(clCommandQueue, input, CL_TRUE, 
                          y*img_width*sizeof(unsigned int), // Offset in bytes
                          img_width*sizeof(unsigned int), // Size of data to write
                          &img[y*img_width], // Host Source
                          0, NULL, NULL) );
  }
 
  pb_SwitchToSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);

  unsigned int img_dim = img_height*img_width;
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_intermediates_kernel, 0, sizeof(cl_mem), (void *)&input) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_intermediates_kernel, 1, sizeof(unsigned int), &img_width) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_intermediates_kernel, 2, sizeof(cl_mem), (void *)&global_subhisto) );
  
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 0, sizeof(unsigned int), &histo_height) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 1, sizeof(unsigned int), &histo_width) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 2, sizeof(cl_mem), (void *)&global_subhisto) );
  OCL_ERRCK_RETVAL( clSetKernelArg(histo_final_kernel, 3, sizeof(cl_mem), (void *)&final_histo) );

  size_t inter_localWS[1] = { workItemSizes[0] };
  size_t inter_globalWS[1] = { img_height * inter_localWS[0] };
  
  size_t final_localWS[1] = { workItemSizes[0] };
  size_t final_globalWS[1] = {((histo_height*histo_width+(final_localWS[0]-1)) /
                                          final_localWS[0])*final_localWS[0] };
  
  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

  for (int iter = 0; iter < numIterations; iter++) {
    unsigned int ranges_h[2] = {UINT32_MAX, 0};
    
    // how about something like
    // __global__ unsigned int ranges[2];
    // ...kernel
    // __shared__ unsigned int s_ranges[2];
    // if (threadIdx.x == 0) {s_ranges[0] = ranges[0]; s_ranges[1] = ranges[1];}
    // __syncthreads();
    
    // Although then removing the blocking cudaMemcpy's might cause something about
    // concurrent kernel execution.
    // If kernel launches are synchronous, then how can 2 kernels run concurrently? different host threads?


  OCL_ERRCK_RETVAL( clEnqueueWriteBuffer(clCommandQueue, ranges, CL_TRUE, 
                          0, // Offset in bytes
                          2*sizeof(unsigned int), // Size of data to write
                          ranges_h, // Host Source
                          0, NULL, NULL) );
                          
  OCL_ERRCK_RETVAL( clEnqueueWriteBuffer(clCommandQueue, global_subhisto, CL_TRUE, 
                          0, // Offset in bytes
                          histo_width*histo_height*sizeof(unsigned int), // Size of data to write
                          zeroData, // Host Source
                          0, NULL, NULL) );
                          
  pb_SwitchToSubTimer(&timers, intermediates, pb_TimerID_KERNEL);

  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, histo_intermediates_kernel /*histo_intermediates_kernel*/, 1, 0,
                            inter_globalWS, inter_localWS, 0, 0, 0) );              
  pb_SwitchToSubTimer(&timers, finals, pb_TimerID_KERNEL);                            
  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, histo_final_kernel, 1, 0,
                            final_globalWS, final_localWS, 0, 0, 0) );                           
  }

  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  OCL_ERRCK_RETVAL( clEnqueueReadBuffer(clCommandQueue, final_histo, CL_TRUE, 
                          0, // Offset in bytes
                          histo_height*histo_width*sizeof(unsigned char), // Size of data to read
                          histo, // Host Source
                          0, NULL, NULL) );                         

  OCL_ERRCK_RETVAL ( clReleaseKernel(histo_intermediates_kernel) );
  OCL_ERRCK_RETVAL ( clReleaseKernel(histo_final_kernel) );
  OCL_ERRCK_RETVAL ( clReleaseProgram(clProgram[0]) );
  OCL_ERRCK_RETVAL ( clReleaseProgram(clProgram[1]) );
  
  OCL_ERRCK_RETVAL ( clReleaseMemObject(input) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(ranges) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(sm_mappings) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(global_subhisto) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(global_overflow) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(final_histo) );

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(zeroData);
  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);
  
  pb_DestroyTimerSet(&timers);

  OCL_ERRCK_RETVAL ( clReleaseCommandQueue(clCommandQueue) );
  OCL_ERRCK_RETVAL ( clReleaseContext(clContext) );

  return 0;
}
