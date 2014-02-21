#include "OpenCL_common.h"
#include <stdlib.h>
#include <string.h>


int getOpenCLPlatform(cl_platform_id *platform, int numPlatformRequests, ...) {

}

bool checkDeviceReq(cl_device_id clDevice, cl_device_info request, void *value) {
  
  bool final = false;
  size_t retValSize;
  
  size_t var_sizet, d_sizet;
  cl_ulong var_ulong, d_ulong;
  cl_uint var_uint, d_uint;
  cl_device_type device_type, d_device_type;
  cl_bool var_bool, d_bool;
  
  cl_platform_id platform_id, d_platform_id;
  cl_device_exec_capabilities exec_capabilities, d_exec_capabilities;
  cl_command_queue_properties command_queue_properties, d_command_queue_properties;
  cl_device_mem_cache_type mem_cache_type, d_mem_cache_type;
  cl_device_local_mem_type local_mem_type, d_local_mem_type;
  cl_device_fp_config fp_config, d_fp_config;

  size_t var_sizes[8], d_var_sizes[8]; // assumes less than 8-D
  char var_string[1024], d_string[1024];
  
  switch (request) {
    case CL_DEVICE_ADDRESS_BITS:
    case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
    case CL_DEVICE_MAX_CLOCK_FREQUENCY:
    case CL_DEVICE_MAX_COMPUTE_UNITS:
    case CL_DEVICE_MAX_CONSTANT_ARGS:
    case CL_DEVICE_MAX_READ_IMAGE_ARGS:
    case CL_DEVICE_MAX_SAMPLERS:
    case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
    case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
    case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
    case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
    case CL_DEVICE_VENDOR_ID:
           var_uint = *((cl_uint *)value);
           retValSize = sizeof(cl_uint);
           OCL_SIMPLE_ERRCK_RETVAL( clGetDeviceInfo(clDevice, request, retValSize, &d_uint, NULL));
           if (((signed int) var_uint) < 0) {
             if (d_uint < -1*((signed int)var_uint)) {
               final = true;
             }
           } else {
             if (d_uint > var_uint) {
               final = true;
             }
           }
           break;
    case CL_DEVICE_AVAILABLE:
    case CL_DEVICE_COMPILER_AVAILABLE:
    case CL_DEVICE_ENDIAN_LITTLE:
    case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
    case CL_DEVICE_HOST_UNIFIED_MEMORY:
    case CL_DEVICE_IMAGE_SUPPORT:
           var_bool = *((cl_bool *)value);
           retValSize = sizeof(cl_bool);
           OCL_SIMPLE_ERRCK_RETVAL( clGetDeviceInfo(clDevice, request, retValSize, &d_bool, NULL));
           if (d_bool == var_bool) {
             final = true;
           }
           break;
#if ( __OPENCL_VERSION__ >= CL_VERSION_1_1 )
    case CL_DEVICE_DOUBLE_FP_CONFIG:
    case CL_DEVICE_HALF_FP_CONFIG:
#endif
    case CL_DEVICE_SINGLE_FP_CONFIG:
           fp_config = *((cl_device_fp_config *)value);
           retValSize = sizeof(cl_device_fp_config);
           OCL_SIMPLE_ERRCK_RETVAL( clGetDeviceInfo(clDevice, request, retValSize, &d_fp_config, NULL));
           if (d_fp_config & fp_config) {
             final = true;
           }
           
           
           
  }
}

// -1 for NO suitable device found, 0 if an appropriate device was found
int getOpenCLDevice(cl_platform_id *platform, cl_device_id *device, cl_device_type *reqDeviceType, int numDeviceRequests, ...) {
      
      // Sequence of (cl_device_type_info, desired value) pairs
      // for example:
      // ..., CL_DEVICE_IMAGE_SUPPORT, true, CL_DEVICE_GLOBAL_MEM_SIZE, 12345678, ...
      // tests ">=" for numerical values.  If "<" is desired, then pass as a negative value of the same type.  However, this may not always work.
      // for example, "less than UINT32_MAX" is not yet supported since the cl_device_info triggers type "UINT32" -- this may be implemented later
      
      // For booleans, pass 'true' or 'false' -- *do not pass integers!!*????
  
  cl_uint numEntries = 16;
  cl_platform_id clPlatforms[numEntries];
  cl_uint numPlatforms;
  
  cl_device_id clDevices[numEntries];
  cl_uint numDevices;

  OCL_SIMPLE_ERRCK_RETVAL ( clGetPlatformIDs(numEntries, clPlatforms, &numPlatforms) );
  fprintf(stderr, "Number of Platforms found: %d\n", numPlatforms);
  bool needDevice = true;
  
  cl_device_info deviceRequests[numDeviceRequests];
  
  for (int ip = 0; ip < numPlatforms && needDevice; ++ip) {

    cl_platform_id clPlatform = clPlatforms[ip];
    
    OCL_SIMPLE_ERRCK_RETVAL ( clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, numEntries, clDevices, &numDevices) );
    fprintf(stderr, "  Number of Devices found for Platform %d: %d\n", ip, numDevices);
    
    for (int id = 0; (id < numDevices) && needDevice ; ++id) {
      cl_device_id clDevice = clDevices[id];
      cl_device_type clDeviceType;

      bool canSatisfy = true;
      
      if (reqDeviceType != NULL) {
        OCL_SIMPLE_ERRCK_RETVAL( clGetDeviceInfo(clDevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &clDeviceType, NULL));
        if (*reqDeviceType != CL_DEVICE_TYPE_ALL) {
          if (*reqDeviceType != clDeviceType) {
            canSatisfy = false;
          }
        }
      }

      va_list paramList;
      va_start(paramList, numDeviceRequests);
      for (int i = 0; (i < numDeviceRequests) && canSatisfy ; ++i) {
      
        cl_device_info devReq = va_arg( paramList, cl_device_info );  
        cl_bool clInfoBool;
        size_t infoRetSize = sizeof(cl_bool);
        
        OCL_SIMPLE_ERRCK_RETVAL( clGetDeviceInfo(clDevice, devReq, infoRetSize, &clInfoBool, NULL));
        if (clInfoBool != true) {
          canSatisfy = false;
        }
      }
      
      va_end(paramList);
      if (canSatisfy) {
        *device = clDevice;
        *platform = clPlatform;
        needDevice = false;
        fprintf(stderr, "Chose Device Type: %s\n",
          (clDeviceType == CL_DEVICE_TYPE_CPU) ? "CPU" : (clDeviceType == CL_DEVICE_TYPE_GPU) ? "GPU" : "other"
          );
        if (reqDeviceType != NULL && (*reqDeviceType == CL_DEVICE_TYPE_ALL)) {
          *reqDeviceType = clDeviceType;
        }
      }
    } // End checking all devices for a platform
  } // End checking all platforms

  int retVal = -1;
  if (needDevice) {
    retVal = -1;
  } else {
    retVal = 0;
  }
  
  return retVal;
}

const char* oclErrorString(cl_int error)
{
// From NVIDIA SDK
	static const char* errorString[] = {
		"CL_SUCCESS",
		"CL_DEVICE_NOT_FOUND",
		"CL_DEVICE_NOT_AVAILABLE",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_MEM_OBJECT_ALLOCATION_FAILURE",
		"CL_OUT_OF_RESOURCES",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_PROFILING_INFO_NOT_AVAILABLE",
		"CL_MEM_COPY_OVERLAP",
		"CL_IMAGE_FORMAT_MISMATCH",
		"CL_IMAGE_FORMAT_NOT_SUPPORTED",
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_MAP_FAILURE",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"CL_INVALID_VALUE",
		"CL_INVALID_DEVICE_TYPE",
		"CL_INVALID_PLATFORM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_CONTEXT",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_INVALID_COMMAND_QUEUE",
		"CL_INVALID_HOST_PTR",
		"CL_INVALID_MEM_OBJECT",
		"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE",
		"CL_INVALID_SAMPLER",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION",
		"CL_INVALID_KERNEL",
		"CL_INVALID_ARG_INDEX",
		"CL_INVALID_ARG_VALUE",
		"CL_INVALID_ARG_SIZE",
		"CL_INVALID_KERNEL_ARGS",
		"CL_INVALID_WORK_DIMENSION",
		"CL_INVALID_WORK_GROUP_SIZE",
		"CL_INVALID_WORK_ITEM_SIZE",
		"CL_INVALID_GLOBAL_OFFSET",
		"CL_INVALID_EVENT_WAIT_LIST",
		"CL_INVALID_EVENT",
		"CL_INVALID_OPERATION",
		"CL_INVALID_GL_OBJECT",
		"CL_INVALID_BUFFER_SIZE",
		"CL_INVALID_MIP_LEVEL",
		"CL_INVALID_GLOBAL_WORK_SIZE",
	};

	const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

	const int index = -error;

	return (index >= 0 && index < errorCount) ? errorString[index] : "";
}

const char* oclDebugErrString(cl_int error, cl_device_id device, int verbose,.../*FILE *stream*/)
{

  const char *errorType = NULL;
  const char *verboseMsg = NULL;
  
  FILE *fp = stderr;
  if (verbose > 0) {
    va_list vl;
    va_start(vl, verbose);
    fp = va_arg(vl, FILE* );
    va_end(vl);
  }

  switch (error) {
    case CL_SUCCESS: errorType = "CL_SUCCESS"; break;
    case CL_DEVICE_NOT_FOUND: errorType = "CL_DEVICE_NOT_FOUND"; break;
    case CL_DEVICE_NOT_AVAILABLE: errorType = "CL_DEVICE_NOT_AVAILABLE"; break;
    case CL_COMPILER_NOT_AVAILABLE: errorType = "CL_COMPILER_NOT_AVAILABLE"; break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: errorType = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
    case CL_OUT_OF_RESOURCES: errorType = "CL_OUT_OF_RESOURCES"; break;
    case CL_OUT_OF_HOST_MEMORY: errorType = "CL_OUT_OF_HOST_MEMORY"; break;
    case CL_PROFILING_INFO_NOT_AVAILABLE: errorType = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
    case CL_MEM_COPY_OVERLAP: errorType = "CL_MEM_COPY_OVERLAP"; break;
    case CL_IMAGE_FORMAT_MISMATCH: errorType = "CL_IMAGE_FORMAT_MISMATCH"; break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: errorType = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
    case CL_BUILD_PROGRAM_FAILURE: errorType = "CL_BUILD_PROGRAM_FAILURE"; break;
    case CL_MAP_FAILURE: errorType = "CL_MAP_FAILURE"; break;
    case CL_INVALID_VALUE: errorType = "CL_INVALID_VALUE"; break;
    case CL_INVALID_DEVICE_TYPE: errorType = "CL_INVALID_DEVICE_TYPE"; break;
    case CL_INVALID_PLATFORM: errorType = "CL_INVALID_PLATFORM"; break;
    case CL_INVALID_DEVICE: errorType = "CL_INVALID_DEVICE"; break;
    case CL_INVALID_CONTEXT: errorType = "CL_INVALID_CONTEXT"; break;

    case CL_INVALID_QUEUE_PROPERTIES: errorType = "CL_INVALID_QUEUE_PROPERTIES"; break;
    case CL_INVALID_COMMAND_QUEUE: errorType = "CL_INVALID_COMMAND_QUEUE"; break;
    // Note: this error often results from out-of-bounds memory accesses
    case CL_INVALID_HOST_PTR: errorType = "CL_INVALID_HOST_PTR"; break;
    case CL_INVALID_MEM_OBJECT: errorType = "CL_INVALID_MEM_OBJECT"; break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: errorType = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
    case CL_INVALID_IMAGE_SIZE: errorType = "CL_INVALID_IMAGE_SIZE"; break;
    case CL_INVALID_SAMPLER: errorType = "CL_INVALID_SAMPLER"; break;
    case CL_INVALID_BINARY: errorType = "CL_INVALID_BINARY"; break;
    case CL_INVALID_BUILD_OPTIONS: errorType = "CL_INVALID_BUILD_OPTIONS"; break;
    // '-D name', '-D name=def', '-I dir', '-cl-single-precision-constant', '-cl-denorms-are-zero', '-cl-opt-disable', '-cl-mad-enable', '-cl-no-signed-zeros', '-cl-unsafe-math-optimizations', '-cl-finite-math-only', '-cl-fast-relaxed-math', '-w', '-Werror', '-cl-std=CL1.1
    case CL_INVALID_PROGRAM: errorType = "CL_INVALID_PROGRAM"; break;
    case CL_INVALID_PROGRAM_EXECUTABLE: errorType = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
    case CL_INVALID_KERNEL: errorType = "CL_INVALID_KERNEL"; break;
    case CL_INVALID_KERNEL_NAME: errorType = "CL_INVALID_KERNEL_NAME"; break;
    case CL_INVALID_ARG_INDEX: errorType = "CL_INVALID_ARG_INDEX"; break;
    case CL_INVALID_ARG_VALUE: errorType = "CL_INVALID_ARG_VALUE"; break;
    case CL_INVALID_ARG_SIZE: errorType = "CL_INVALID_ARG_SIZE"; break;
    case CL_INVALID_KERNEL_ARGS: errorType = "CL_INVALID_KERNEL_ARGS"; break;
    case CL_INVALID_WORK_DIMENSION: errorType = "CL_INVALID_WORK_DIMENSION"; break;
    case CL_INVALID_WORK_GROUP_SIZE: errorType = "CL_INVALID_WORK_GROUP_SIZE"; break;
    case CL_INVALID_WORK_ITEM_SIZE: errorType = "CL_INVALID_WORK_ITEM_SIZE"; break;
    case CL_INVALID_GLOBAL_OFFSET: errorType = "CL_INVALID_GLOBAL_OFFSET"; break;
    case CL_INVALID_EVENT_WAIT_LIST: errorType = "CL_INVALID_EVENT_WAIT_LIST"; break;
    case CL_INVALID_EVENT: errorType = "CL_INVALID_EVENT"; break;
    case CL_INVALID_OPERATION: errorType = "CL_INVALID_OPERATION"; break;
    case CL_INVALID_GL_OBJECT: errorType = "CL_INVALID_GL_OBJECT"; break;
    case CL_INVALID_BUFFER_SIZE: errorType = "CL_INVALID_BUFFER_SIZE"; break;
    case CL_INVALID_MIP_LEVEL: errorType = "CL_INVALID_MIP_LEVEL"; break;
    case CL_INVALID_GLOBAL_WORK_SIZE: errorType = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
    
    // May not be in OpenCL 1.0
#ifdef CL_INVALID_PROPERTY
    case CL_INVALID_PROPERTY: errorType = "CL_INVALID_PROPERTY"; break;
#endif
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: errorType = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: errorType = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
    default: errorType = "Error code not found!";
  }
  
  if (verbose) {
    cl_ulong var_ulong = 0;
    switch (error) {
      case CL_INVALID_COMMAND_QUEUE:
             fprintf(fp, "  Note: this error often results from out-of-bounds memory accesses\n");
             break;
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:
             //cl_ulong maxMemAlloc = 0;
	         OCL_SIMPLE_ERRCK_RETVAL ( clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &var_ulong, NULL) );
	         fprintf(fp, "  Device Maximum block allocation size: %lu\n", var_ulong);
             break;
      case CL_INVALID_BUILD_OPTIONS:
             fprintf(fp, "Valid build options include (but not limited to): '-D name', '-D name=def', '-I dir', '-cl-single-precision-constant', '-cl-denorms-are-zero', '-cl-opt-disable', '-cl-mad-enable', '-cl-no-signed-zeros', '-cl-unsafe-math-optimizations', '-cl-finite-math-only', '-cl-fast-relaxed-math', '-w', '-Werror', '-cl-std=CL1.1'\n");
             break;
    }
  }

	return errorType;
}

char* readFile(const char* fileName)
{
        FILE* fp;
        fp = fopen(fileName,"r");
        if(fp == NULL)
        {
                printf("Error 1!\n");
                exit(1);
        }

        fseek(fp,0,SEEK_END);
        long size = ftell(fp);
        rewind(fp);

        char* buffer = (char*)malloc(sizeof(char)*size+1);
        if(buffer  == NULL)
        {
                printf("Error 2!\n");
                fclose(fp);
                exit(1);
        }

        size_t res = fread(buffer,1,size,fp);
        if(res != size)
        {
                printf("Error 3!\n");
                fclose(fp);
                exit(1);
        }
        buffer[size] = '\0';

        fclose(fp);
        return buffer;
}
