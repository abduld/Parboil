
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#define OCL_CPU 0
#if OCL_CPU
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

// Possible values are 2, 4, 8 and 16
#define R 2

#if 0
inline float2 cmpMul( float2 a, float2 b ) {
  float2 result = { a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x };
  return result;
}
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643
#endif

#define COS_PI_8  0.923879533f
#define SIN_PI_8  0.382683432f
#define exp_1_16  (float2) (  COS_PI_8, -SIN_PI_8 )
#define exp_3_16  (float2) (  SIN_PI_8, -COS_PI_8 )
#define exp_5_16  (float2) ( -SIN_PI_8, -COS_PI_8 )
#define exp_7_16  (float2) ( -COS_PI_8, -SIN_PI_8 )
#define exp_9_16  (float2) ( -COS_PI_8,  SIN_PI_8 )
#define exp_1_8   (float2) (  1, -1 )
#define exp_1_4   (float2) (  0, -1 )
#define exp_3_8   (float2) ( -1, -1 )

/*
inline void global_GPU_FFT2(__private float2* v){
  float2 v0 = v[0];
  float2 v3 = v[1];
  v[0] = v0 + v3; 
  v[1] = v0 - v3; 
}
*/

/*
inline void GPU_FFT2(__private float2 *v1, __private float2 *v2 ) { 
  float2 v0 = *v1;
  float2 v3 = *v2;
  *v1 = v0 + v3; 
  *v2 = v0 - v3; 
}

inline void GPU_FFT4(__private float2 *v0, __private float2 *v1, __private float2 *v2, __private float2 *v3) { 
   GPU_FFT2(v0, v2);
   GPU_FFT2(v1, v3);
   *v3 = cmpMul((*v3) , exp_1_4 );
   GPU_FFT2(v0, v1);
   GPU_FFT2(v2, v3);    
}

inline void global_GPU_FFT4(__private float2* v){
  GPU_FFT4(v[0],v[1],v[2],v[3] );
}


inline void global_GPU_FFT8(__private float2* v){
  GPU_FFT2(v[0],v[4]);
  GPU_FFT2(v[1],v[5]);
  GPU_FFT2(v[2],v[6]);
  GPU_FFT2(v[3],v[7]);

  v[5]=(cmpMul(v[5], exp_1_8 ))*M_SQRT1_2;
  v[6]=(cmpMul(v[6], exp_1_4 );
  v[7]=cmpMul((v[7],exp_3_8))*M_SQRT1_2;

  GPU_FFT4(v[0],v[1],v[2],v[3]);
  GPU_FFT4(v[4],v[5],v[6],v[7]);
  
}

inline void global_GPU_FFT16( __private float2 *v )
{
    GPU_FFT4( v[0], v[4], v[8], v[12] );
    GPU_FFT4( v[1], v[5], v[9], v[13] );
    GPU_FFT4( v[2], v[6], v[10], v[14] );
    GPU_FFT4( v[3], v[7], v[11], v[15] );

    v[5]  = cmpMul(v[5]  , exp_1_8 ) * M_SQRT1_2;
    v[6]  = cmpMul( v[6]  , exp_1_4);
    v[7]  = cmpMul(v[7]  , exp_3_8 ) * M_SQRT1_2;
    v[9]  =  cmpMul(v[9]  , exp_1_16);
    v[10] = cmpMul(v[10] , exp_1_8 ) * M_SQRT1_2;
    v[11] =  cmpMul(v[11] , exp_3_16);
    v[13] =  cmpMul(v[13] , exp_3_16);
    v[14] = cmpMul((v[14] , exp_3_8 ) * M_SQRT1_2;
    v[15] =  cmpMul(v[15] , exp_9_16);

    GPU_FFT4( v[0],  v[1],  v[2],  v[3] );
    GPU_FFT4( v[4],  v[5],  v[6],  v[7] );
    GPU_FFT4( v[8],  v[9],  v[10], v[11] );
    GPU_FFT4( v[12], v[13], v[14], v[15] );
}
*/
  
     
int GPU_expand(int idxL, int N1, int N2 ){ 
  return (idxL/N1)*N1*N2 + (idxL%N1); 
}      

__kernel void GPU_FFT_Global(int Ns, __global float* data0, __global float* data1, int N) {
  int bx = get_group_id(0);
  int tx = get_local_id(0);

  data0+=2*bx*N;
  data1+=2*bx*N;
  {
    __private float v_x[R];
    __private float v_y[R];

    int idxS = tx;
    float angle = -2*M_PI*(tx%Ns)/(Ns*R);

    for( int r=0; r<R; r++ ) { 
      v_x[r] = data0[2*(idxS+r*N/R)];
      v_y[r] = data0[2*(idxS+r*N/R)+1];
      float a_x = v_x[r];
      float a_y = v_y[r];
      float b_x, b_y;
      b_x = cos(r*angle); b_y = sin(r*angle);
      float v_r_x = a_x*b_x-a_y*b_y;
      float v_r_y = a_x*b_y+a_y*b_x;
      v_x[r] = v_r_x;
      v_y[r] = v_r_y;
    }
  
  #if R == 2 
  {
    float v0_x = v_x[0];
    float v0_y = v_y[0];
    float v3_x = v_x[1];
    float v3_y = v_y[1];
    v_x[0] = v0_x + v3_x;
    v_y[0] = v0_y + v3_y;
    v_x[1] = v0_x - v3_x;
    v_y[1] = v0_y - v3_y;
  }
  #endif
  
#if 0
  #if R == 4
    global_GPU_FFT4( v );
  #endif	 	
  
  #if R == 8
    global_GPU_FFT8( v );
  #endif
  
  #if R == 16
    global_GPU_FFT16( v );
  #endif	 	
#endif
  
    int idxD = (tx/Ns)*Ns*R + (tx%Ns); 
  
    for( int r=0; r<R; r++ ){
      data1[2*(idxD+r*Ns)] = v_x[r];
      data1[2*(idxD+r*Ns)+1] = v_y[r];
    } 	
  }
}

