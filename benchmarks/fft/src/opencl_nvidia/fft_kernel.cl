/*
Author: Sara Baghsorkhi.
This implementation is partly based on the SC08 paper by Naga K. Govindaraju et al.
*/

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

#if 0
inline float2 cmpMul( float2 a, float2 b ) { return (float2)( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643
#endif

#if 0
#define make_float2(x, y)    ((x), (y))

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

/*
void GPU_FFT2( float2 &v1,float2 &v2 ) { 
  float2 v0 = v1;  
  v1 = v0 + v2; 
  v2 = v0 - v2; 
}
*/
void GPU_FFT2(__private float2 *v1, __private float2 *v2 ) { 
  float2 v0 = *v1;  
  *v1 = v0 + *v2; 
  *v2 = v0 - *v2; 
}

/*
void GPU_FFT4( float2 &v0,float2 &v1,float2 &v2,float2 &v3) { 
   GPU_FFT2(v0, v2);
   GPU_FFT2(v1, v3);
   v3 = v3 * exp_1_4;
   GPU_FFT2(v0, v1);
   GPU_FFT2(v2, v3);    
}
*/
void GPU_FFT4(__private float2 *v0, __private float2 *v1, __private float2 *v2, __private float2 *v3) { 
   GPU_FFT2(v0, v2);
   GPU_FFT2(v1, v3);
   *v3 = cmpMul(*v3 , exp_1_4);
   GPU_FFT2(v0, v1);
   GPU_FFT2(v2, v3);    
}


inline void ptr_GPU_FFT2(__private float2* v){
  GPU_FFT2(v, v+1);
}

inline void ptr_GPU_FFT4(__private float2* v){
  GPU_FFT4(v, v+1, v+2, v+3 );
}


inline void ptr_GPU_FFT8(__private float2* v){
  GPU_FFT2(v, v+4);
  GPU_FFT2(v+1, v+5);
  GPU_FFT2(v+2, v+6);
  GPU_FFT2(v+3, v+7);

  v[5]=cmpMul(v[5], exp_1_8)*M_SQRT1_2;
  v[6]=cmpMul(v[6], exp_1_4);
  v[7]=cmpMul(v[7], exp_3_8)*M_SQRT1_2;


  GPU_FFT4(v,v+1,v+2,v+3);
  GPU_FFT4(v+4,v+5,v+6,v+7);
  
}


inline void ptr_GPU_FFT16(__private float2 *v ) {
  GPU_FFT4(v,v+4,v+8,v+12);
  GPU_FFT4(v+1,v+5,v+9,v+13);
  GPU_FFT4(v+2,v+6,v+10,v+14);
  GPU_FFT4(v+3,v+7,v+11,v+15);

    v[5]  = cmpMul(v[5] , exp_1_8 ) * M_SQRT1_2;
    v[6]  = cmpMul(v[6] , exp_1_4 );
    v[7]  = cmpMul(v[7] , exp_3_8 ) * M_SQRT1_2;
    v[9]  = cmpMul(v[9] , exp_1_16);
    v[10] = cmpMul(v[10] , exp_1_8 ) * M_SQRT1_2;
    v[11] = cmpMul(v[11] , exp_3_16);
    v[13] = cmpMul(v[13] , exp_3_16);
    v[14] = cmpMul(v[14] , exp_3_8 ) * M_SQRT1_2;
    v[15] = cmpMul(v[15] , exp_9_16);

  GPU_FFT4(v,v+1,v+2,v+3);
  GPU_FFT4(v+4,v+5,v+6,v+7);
  GPU_FFT4(v+8,v+9,v+10,v+11);
  GPU_FFT4(v+12,v+13,v+14,v+15);
}
     
int GPU_expand(int idxL, int N1, int N2 ){ 
  return (idxL/N1)*N1*N2 + (idxL%N1); 
}      

void GPU_exchange(__private float2* v, int stride, int idxD, int incD, 
	int idxS, int incS ){ 
  __local float work[T*R*2];//T*R*2
  __local float* sr = work;
  __local float* si = work+T*R;  
  
  barrier( CLK_LOCAL_MEM_FENCE );
  for( int r=0; r<R; r++ ) { 
    int i = (idxD + r*incD)*stride; 
    sr[i] = v[r].x;
    si[i] = v[r].y;  
  }
  barrier( CLK_LOCAL_MEM_FENCE );

  for( int r=0; r<R; r++ ) { 
    int i = (idxS + r*incS)*stride;     
    v[r] = (float2) (sr[i], si[i]);  
  }        
}      

  
void GPU_DoFft(__private float2* v, int j) {

  const int stride=1;
  for( int Ns=1; Ns<N; Ns*=R ){ 
    float angle = -2*M_PI*(j%Ns)/(Ns*R); 
    for( int r=0; r<R; r++ ){
      v[r] = cmpMul(v[r], (float2) (cos(r*angle), sin(r*angle)));
    }
#if R2
    ptr_GPU_FFT2( v );
#endif

#if R4
    ptr_GPU_FFT4( v );
#endif

#if R8
    ptr_GPU_FFT8( v );	
#endif

#if R16
    ptr_GPU_FFT16( v );
#endif

    int idxD = GPU_expand(j,Ns,R); 
    int idxS = GPU_expand(j,N/R,R); 
    GPU_exchange( v,stride, idxD,Ns, idxS,N/R );
  }      
}

#endif

__kernel void GPU_FftShMem(__global float* data){ 
  int bx = get_group_id(0);
  int tx = get_local_id(0);

  float v_x[R];
  float v_y[R];
  data+=bx*N*2;

  int idxG = tx; 
  for( int r=0; r<R; r++ ){  
    v_x[r] = data[2*(idxG + r*T)];
    v_y[r] = data[2*(idxG + r*T)+1];
  } 

  // GPU_DoFft( v, tx );  
  {
    const int stride=1;
    for( int Ns=1; Ns<N; Ns*=R ) {
      float angle = -2*M_PI*(tx%Ns)/(Ns*R);
      for( int r=0; r<R; r++ ){
        float a_x = v_x[r];
        float a_y = v_y[r];
        float b_x, b_y;
        b_x = cos(r*angle); b_y = sin(r*angle);
        float v_r_x = a_x*b_x-a_y*b_y;
        float v_r_y = a_x*b_y+a_y*b_x;
        v_x[r] = v_r_x;
        v_y[r] = v_r_y;
      }
  #if R2
    #if 0
      ptr_GPU_FFT2( v );
    #else
      {
        float v0_x;
        float v0_y;
        float* v1_x;
        float* v1_y;
        float* v2_x;
        float* v2_y;
        v0_x = (float) tx;
        v0_y = (float) tx;

        v1_x = &v_x[0];
        v1_y = &v_y[0];
        v2_x = &v_x[1];
        v2_y = &v_y[1];
        v0_x = *v1_x;
        v0_y = *v1_y;
        *v1_x = v0_x + *v2_x;
        *v1_y = v0_y + *v2_y;
        *v2_x = v0_x - *v2_x;
        *v2_y = v0_y - *v2_y;
      }
    #endif
  #endif

  #if R4
      ptr_GPU_FFT4( v );
  #endif

  #if R8
      ptr_GPU_FFT8( v );	
  #endif

  #if R16
      ptr_GPU_FFT16( v );
  #endif

      int idxD = (tx/Ns)*Ns*R + (tx%Ns);
      int idxS = (tx/(N/R))*(N/R)*R + (tx%(N/R));
      {
        int incD = Ns;
        int incS = N/R;
        __local float work[T*R*2];//T*R*2
        __local float* sr = work;
        __local float* si = work+T*R;

        barrier( CLK_LOCAL_MEM_FENCE );
        for( int r=0; r<R; r++ ) {
          int i = (idxD + r*incD)*stride;
          sr[i] = v_x[r];
          si[i] = v_y[r];
        }
        barrier( CLK_LOCAL_MEM_FENCE );

        for( int r=0; r<R; r++ ) {
          int i = (idxS + r*incS)*stride;
          v_x[r] = sr[i];
          v_y[r] = si[i];
        }
      }
    }
  }

  for( int r=0; r<R; r++ ) {
    data[2*(idxG + r*T)] = v_x[r];
    data[2*(idxG + r*T)+1] = v_y[r];
  }
}
