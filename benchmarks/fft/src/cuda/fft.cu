/*
Author: Sara Baghsorkhi.
This implementation is partly based on the SC08 paper by Naga K. Govindaraju et al.
*/


#include <stdio.h>
#include <cuda.h>
#include <parboil.h>

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}

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


inline float2 operator*( float2 a, float2 b ) { return make_float2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline float2 operator+( float2 a, float2 b ) { return make_float2( a.x + b.x, a.y + b.y ); }
inline float2 operator-( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }
inline float2 operator*( float2 a, float b ) { return make_float2( b*a.x , b*a.y); }

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
  
void FFT2( float2* v ) { 
  float2 v0 = v[0];  
  v[0] = v0 + v[1]; 
  v[1] = v0 - v[1]; 
}        


__device__ void GPU_FFT2( float2 &v1,float2 &v2 ) { 
  float2 v0 = v1;  
  v1 = v0 + v2; 
  v2 = v0 - v2; 
}

__device__ void GPU_FFT4( float2 &v0,float2 &v1,float2 &v2,float2 &v3) { 
   GPU_FFT2(v0, v2);
   GPU_FFT2(v1, v3);
   v3 = v3 * exp_1_4;
   GPU_FFT2(v0, v1);
   GPU_FFT2(v2, v3);    
}


inline __device__ void GPU_FFT2(float2* v){
  GPU_FFT2(v[0],v[1]);
}

inline __device__ void GPU_FFT4(float2* v){
  GPU_FFT4(v[0],v[1],v[2],v[3] );
}


inline __device__ void GPU_FFT8(float2* v){
  GPU_FFT2(v[0],v[4]);
  GPU_FFT2(v[1],v[5]);
  GPU_FFT2(v[2],v[6]);
  GPU_FFT2(v[3],v[7]);

  v[5]=(v[5]*exp_1_8)*M_SQRT1_2;
  v[6]=v[6]*exp_1_4;
  v[7]=(v[7]*exp_3_8)*M_SQRT1_2;

  GPU_FFT4(v[0],v[1],v[2],v[3]);
  GPU_FFT4(v[4],v[5],v[6],v[7]);
  
}


inline __device__ void GPU_FFT16( float2 *v )
{
    GPU_FFT4( v[0], v[4], v[8], v[12] );
    GPU_FFT4( v[1], v[5], v[9], v[13] );
    GPU_FFT4( v[2], v[6], v[10], v[14] );
    GPU_FFT4( v[3], v[7], v[11], v[15] );

    v[5]  = (v[5]  * exp_1_8 ) * M_SQRT1_2;
    v[6]  =  v[6]  * exp_1_4;
    v[7]  = (v[7]  * exp_3_8 ) * M_SQRT1_2;
    v[9]  =  v[9]  * exp_1_16;
    v[10] = (v[10] * exp_1_8 ) * M_SQRT1_2;
    v[11] =  v[11] * exp_3_16;
    v[13] =  v[13] * exp_3_16;
    v[14] = (v[14] * exp_3_8 ) * M_SQRT1_2;
    v[15] =  v[15] * exp_9_16;

    GPU_FFT4( v[0],  v[1],  v[2],  v[3] );
    GPU_FFT4( v[4],  v[5],  v[6],  v[7] );
    GPU_FFT4( v[8],  v[9],  v[10], v[11] );
    GPU_FFT4( v[12], v[13], v[14], v[15] );
}
     
__device__ int GPU_expand(int idxL, int N1, int N2 ){ 
  return (idxL/N1)*N1*N2 + (idxL%N1); 
}      

__device__ void GPU_exchange( float2* v, int stride, int idxD, int incD, 
	int idxS, int incS ){ 
  __shared__ float work[T*R*2];//T*R*2
  float* sr = work;
  float* si = work+T*R;  
  __syncthreads(); 
  for( int r=0; r<R; r++ ) { 
    int i = (idxD + r*incD)*stride; 
    sr[i] = v[r].x;
    si[i] = v[r].y;  
  }   
  __syncthreads(); 

  for( int r=0; r<R; r++ ) { 
    int i = (idxS + r*incS)*stride;     
    v[r] = make_float2(sr[i], si[i]);  
  }        
}      

  
__device__ void GPU_DoFft(float2* v, int j, int stride=1) { 
  for( int Ns=1; Ns<N; Ns*=R ){ 
    float angle = -2*M_PI*(j%Ns)/(Ns*R); 
    for( int r=0; r<R; r++ ){
      v[r] = v[r]*make_float2(cos(r*angle), sin(r*angle));
    }
#if R2
    GPU_FFT2( v );
#endif

#if R4
    GPU_FFT4( v );
#endif

#if R8
    GPU_FFT8( v );	
#endif


#if R16
    GPU_FFT16( v );
#endif

    int idxD = GPU_expand(j,Ns,R); 
    int idxS = GPU_expand(j,N/R,R); 
    GPU_exchange( v,stride, idxD,Ns, idxS,N/R );
  }      
}    


__global__ void GPU_FftShMem(float2* data){ 
  float2 v[R];
  data+=bx*N; 
  	
  int idxG = tx; 
  for( int r=0; r<R; r++ ){  
    v[r] = data[idxG + r*T];
  } 
  GPU_DoFft( v, tx );  
  for( int r=0; r<R; r++ )  
    data[idxG + r*T] = v[r]; 
}    

int main( int argc, char **argv )
{	
  //8*1024*1024;
  int n_bytes = N * B* sizeof(float2);
  int nthreads = T;
  srand(54321);
  
  struct pb_TimerSet timers;
  pb_InitializeTimerSet(&timers);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  float *shared_source =(float *)malloc(n_bytes);  
  float2 *source    = (float2 *)malloc( n_bytes );
  float2 *result    = (float2 *)malloc( n_bytes );

  for(int b=0; b<B;b++){	
    for( int i = 0; i < N; i++ ){
      source[b*N+i].x = (rand()/(float)RAND_MAX)*2-1;
      source[b*N+i].y = (rand()/(float)RAND_MAX)*2-1;
    }
  }

  // allocate device memory
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);
  float2 *d_source, *d_work;
  float *d_shared_source;
  cudaMalloc((void**) &d_shared_source, n_bytes);
  CUERR;
  // copy host memory to device
  cudaMemcpy(d_shared_source, shared_source, n_bytes,cudaMemcpyHostToDevice);
  CUERR;
  cudaMalloc((void**) &d_source, n_bytes);
  CUERR;
  // copy host memory to device
  cudaMemcpy(d_source, source, n_bytes,cudaMemcpyHostToDevice);
  CUERR;
  cudaMalloc((void**) &d_work, n_bytes);
  CUERR;
  cudaMemset(d_work, 0,n_bytes);
  CUERR;

  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    GPU_FftShMem<<<dim3(B), dim3(nthreads)>>>(d_source); 	
  pb_SwitchToTimer(&timers, pb_TimerID_COPY);


    // copy device memory to host
    cudaMemcpy(result, d_source, n_bytes,cudaMemcpyDeviceToHost);
    CUERR;
  
  cudaFree(d_source);
  CUERR;
  cudaFree(d_work);
  CUERR;	
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(shared_source);  
  free(source);
  free(result);
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);

}

