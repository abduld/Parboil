/*
Author: Sara Baghsorkhi.
This implementation is partly based on the SC08 paper by Naga K. Govindaraju et al.
*/


#include <stdio.h>
#include <stdlib.h>
#include "OpenCL_common.h"
#include <parboil.h>
#include "file.h"

// Block index
#define  bx  blockIdx.x
#define  by  blockIdx.y
// Thread index
#define tx  threadIdx.x

#define DEBUG 0
#define B 1024 
#define EMUL 0

#define R2 1
#define R4 0
#define R8 0
#define R16 0

#if R2
#define N 4*4*4*4
#define R 2
#endif

#if R4
#define N 4*4*4*4
#define R 4
#endif

#if R8
#define N
#define R 8
#endif

#if R16
#define N 4*4*4*4
#define R 16
#endif

#define T  N/R 

/*
inline float2 operator*( float2 a, float2 b ) { return make_float2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline float2 operator+( float2 a, float2 b ) { return make_float2( a.x + b.x, a.y + b.y ); }
inline float2 operator-( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }
inline float2 operator*( float2 a, float b ) { return make_float2( b*a.x , b*a.y); }
*/

#define COS_PI_8  0.923879533f
#define SIN_PI_8  0.382683432f
#define exp_1_16  make_float2(  COS_PI_8, -SIN_PI_8 )
#define exp_3_16  make_float2(  SIN_PI_8, -COS_PI_8 )
#define exp_5_16  make_float2( -SIN_PI_8, -COS_PI_8 )
#define exp_7_16  make_float2( -COS_PI_8, -SIN_PI_8 )
#define exp_9_16  make_float2( -COS_PI_8,  SIN_PI_8 )
#define exp_1_8   make_float2(  1, -1 )//requires post-multiply by 1/sqrt(2)
#define exp_1_4   make_float2(  0, -1 )
#define exp_3_8   make_float2( -1, -1 )//requires post-multiply by 1/sqrt(2)
  
  /*
void FFT2( float2* v ) { 
  float2 v0 = v[0];  
  v[0] = v0 + v[1]; 
  v[1] = v0 - v[1]; 
}
*/

typedef struct { float x; float y; } float2;
char oclOverhead[] = "OpenCL Overhead";

#undef N
#undef B

namespace {
int N;
int B;
};

int main( int argc, char **argv )
{	
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

  //8*1024*1024;
  int n_bytes = N * B* sizeof(float2);
  int nthreads = T;
  
  struct pb_TimerSet timers;
  pb_InitializeTimerSet(&timers);
  pb_AddSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  float *shared_source =(float *)malloc(n_bytes);  
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

  cl_int clStatus;
  cl_device_id clDevice = (cl_device_id) pb_context->clDeviceId;
  cl_platform_id clPlatform = (cl_platform_id) pb_context->clPlatformId;
  cl_context clContext = (cl_context) pb_context->clContext;
  cl_command_queue clCommandQueue;

  cl_program clProgram;

  cl_kernel fft_kernel;

  cl_mem d_source, d_work; //float2 *d_source, *d_work;
  cl_mem d_shared_source; //float *d_shared_source;

  clCommandQueue = clCreateCommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &clErrNum);
  OCL_ERRCK_VAR(clErrNum);
  
  pb_SetOpenCL(&clContext, &clCommandQueue);
  pb_SwitchToSubTimer(&timers, oclOverhead, pb_TimerID_KERNEL);

  const char *source_path = "src/opencl_nvidia/fft_kernel.cl";
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

  fft_kernel = clCreateKernel(clProgram, "GPU_FftShMem", &clErrNum);
  OCL_ERRCK_VAR(clErrNum);

  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  // allocate & copy device memory
  d_shared_source = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, n_bytes, shared_source, &clErrNum);  OCL_ERRCK_VAR(clErrNum);

  d_source = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, n_bytes, source, &clErrNum);  OCL_ERRCK_VAR(clErrNum);
  
  //result is initially zero'd out
  d_work = clCreateBuffer(clContext, CL_MEM_COPY_HOST_PTR, n_bytes, result, &clErrNum);  OCL_ERRCK_VAR(clErrNum);

  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
  
  size_t block[1] = { nthreads };
  size_t grid[1] = { B*block[0] };
  
  OCL_ERRCK_RETVAL( clSetKernelArg(fft_kernel, 0, sizeof(cl_mem), (void *)&d_source) );
  OCL_ERRCK_RETVAL ( clEnqueueNDRangeKernel(clCommandQueue, fft_kernel, 1, 0,
                            grid, block, 0, 0, 0) ); 	
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);

  // copy device memory to host
  OCL_ERRCK_RETVAL( clEnqueueReadBuffer(clCommandQueue, d_source, CL_TRUE, 
                        0, // Offset in bytes
                        n_bytes, // Size of data to read
                        result, // Host Source
                        0, NULL, NULL) );
                        
  if (params->outFile) {
    /* Write result to file */
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    outputData(params->outFile, (float*)result, N*B*2);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  }

  OCL_ERRCK_RETVAL ( clReleaseMemObject(d_source) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(d_work) );
  OCL_ERRCK_RETVAL ( clReleaseMemObject(d_shared_source) );
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(shared_source);  
  free(source);
  free(result);
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  
  pb_DestroyTimerSet(&timers);
  
  return 0;
}

