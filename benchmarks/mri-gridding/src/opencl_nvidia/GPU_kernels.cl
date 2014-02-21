/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#ifndef CUTOFF2_VAL
#define CUTOFF2_VAL 6.250000
#define CUTOFF_VAL 2.500000
#define CEIL_CUTOFF_VAL 3.000000
#define GRIDSIZE_VAL1 256
#define GRIDSIZE_VAL2 256
#define GRIDSIZE_VAL3 256
#define SIZE_XY_VAL 65536
#define ONE_OVER_CUTOFF2_VAL 0.160000
#define DYN_LOCAL_MEM_SIZE 546
#endif

#define TILE 64
#define LOG_TILE 6

typedef struct{
  float real;
  float imag;
  float kX;
  float kY;
  float kZ;
  float sdc;
} ReconstructionSample;

// clear
__kernel void binning_kernel (unsigned int n, 
                              __global ReconstructionSample* sample_g, 
                              __global unsigned int* idxKey_g,
                              __global unsigned int* idxValue_g, 
                              __global unsigned int* binCount_g, 
                              unsigned int binsize, unsigned int gridNumElems){
  unsigned int key;
  unsigned int sampleIdx = get_global_id(0);
  ReconstructionSample pt;
  unsigned int binIdx;
  unsigned int count;

  // promoted due to CEAN bug
  __global unsigned int* p = (__global unsigned int *) get_global_id(0);

  if (sampleIdx < n){
    pt = sample_g[sampleIdx];

    binIdx = (unsigned int)(pt.kZ)*SIZE_XY_VAL + (unsigned int)(pt.kY)*GRIDSIZE_VAL1 + (unsigned int)(pt.kX);
    if (binCount_g[binIdx]<binsize){
      p = binCount_g + binIdx;
      count = atom_add(p, 1);
      if (count < binsize){
        key = binIdx;
      } else {
        atom_sub(p, 1);
        key = gridNumElems;
      }
    } else {
      key = gridNumElems;
    }

    idxKey_g[sampleIdx] = key;
    idxValue_g[sampleIdx] = sampleIdx;
  }
}

// clear
__kernel void reorder_kernel(int n, 
                               __global unsigned int* idxValue_g, 
                               __global ReconstructionSample* samples_g, 
                               __global float* dataptr_g,
                               unsigned int f2_offset
                               ){
  unsigned int index = get_global_id(0);
  unsigned int old_index;
  ReconstructionSample pt;

  // Promoted due to CEAN bug
  float data_x;
  float data_y;
  float loc_x;
  float loc_y;
  float loc_z;
  float loc_w;

  if (index < n){
    old_index = idxValue_g[index];
    pt = samples_g[old_index];

    data_x = pt.real;
    data_y = pt.imag;

    loc_x = pt.kX;
    loc_y = pt.kY;
    loc_z = pt.kZ;
    loc_w = pt.sdc;

    dataptr_g[2*index] = data_x;
    dataptr_g[2*index+1] = data_y;

    ((__global float*)(dataptr_g+2*f2_offset))[4*index  ] = loc_x;
    ((__global float*)(dataptr_g+2*f2_offset))[4*index+1] = loc_y;
    ((__global float*)(dataptr_g+2*f2_offset))[4*index+2] = loc_z;
    ((__global float*)(dataptr_g+2*f2_offset))[4*index+3] = loc_w;
  }
}

// clear
float kernel_value(float v){

  float rValue = 0;

  float z = v*v;

  // polynomials taken from http://ccrma.stanford.edu/CCRMA/Courses/422/projects/kbd/kbdwindow.cpp
#if 0
  float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
                (z*
                 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
                 0.479440257548300e-16f) + 0.435125971262668e-13f ) +
                 0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
                 0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
                 0.463076284721000e0f)   + 0.754337328948189e2f   ) +
                 0.830792541809429e4f)   + 0.571661130563785e6f   ) +
                 0.216415572361227e8f)   + 0.356644482244025e9f   ) +
                 0.144048298227235e10f);

  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);
#else
  float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
                (z* 0.2105e-22f  + 0.3807e-19f ) +
                    0.4799e-16f) + 0.4351e-13f ) +
                    0.3009e-10f) + 0.1602e-7f  ) +
                    0.6548e-5f)  + 0.2025e-2f  ) +
                    0.4630e0f)   + 0.7543e2f   ) +
                    0.8307e4f)   + 0.5716e6f   ) +
                    0.2164e8f)   + 0.3566e9f   ) +
                    0.1440e10f);

  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144e10f);
#endif

  //rValue = native_divide(-num,den);
  //rValue = __fdividef(-num,den);
  rValue = -num / den;

  return rValue;
}

// clear
__kernel void gridding_GPU (  __global float* dataptr_g,
                              unsigned int f2_offset,
                              __global unsigned int* binStartAddr_g, 
                              __global float* gridData_g, 
                              __global float* sampleDensity_g, 
                              float beta){
  __local float real_s[TILE];
  __local float imag_s[TILE];
  __local float kx_s[TILE];
  __local float ky_s[TILE];
  __local float kz_s[TILE];
  __local float sdc_s[TILE];

  const int flatIdx = 
  get_local_id(2)*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0);
  //threadIdx.z*blockDim.y*blockDim.x+threadIdx.y*blockDim.x+threadIdx.x;

  // figure out starting point of the tile
  const int z0 = (4*get_local_size(2))*(get_group_id(1)/(GRIDSIZE_VAL2/get_local_size(1)));
  const int y0 = get_local_size(1)*(get_group_id(1)%(GRIDSIZE_VAL2/get_local_size(1)));
  const int x0 = get_group_id(0)*get_local_size(0);

  const int X  = x0+get_local_id(0);
  const int Y  = y0+get_local_id(1);
  const int Z  = z0+get_local_id(2);
  const int Z1 = Z+get_local_size(2);
  const int Z2 = Z1+get_local_size(2);
  const int Z3 = Z2+get_local_size(2);

  const int xl = x0-CEIL_CUTOFF_VAL;
  const int xL = (xl < 0) ? 0 : xl;
  const int xh = x0+get_local_size(0)+CUTOFF_VAL;
  const int xH = (xh >= GRIDSIZE_VAL1) ? GRIDSIZE_VAL1-1 : xh;

  const int yl = y0-CEIL_CUTOFF_VAL;
  const int yL = (yl < 0) ? 0 : yl;
  const int yh = y0+get_local_size(1)+CUTOFF_VAL;
  const int yH = (yh >= GRIDSIZE_VAL2) ? GRIDSIZE_VAL2-1 : yh;

  const int zl = z0-CEIL_CUTOFF_VAL;
  const int zL = (zl < 0) ? 0 : zl;
  const int zh = z0+(4*get_local_size(2))+CUTOFF_VAL;
  const int zH = (zh >= GRIDSIZE_VAL3) ? GRIDSIZE_VAL3-1 : zh;

  const int idx = Z*SIZE_XY_VAL + Y*GRIDSIZE_VAL1 + X;
  const int idx1 = idx+get_local_size(2)*SIZE_XY_VAL;
  const int idx2 = idx1+get_local_size(2)*SIZE_XY_VAL;
  const int idx3 = idx2+get_local_size(2)*SIZE_XY_VAL;

  float pt_x = 0.0f;
  float pt_y = 0.0f;
  float pt1_x = 0.0f;
  float pt1_y = 0.0f;
  float pt2_x = 0.0f;
  float pt2_y = 0.0f;
  float pt3_x = 0.0f;
  float pt3_y = 0.0f;
  float density = 0.0f;
  float density1 = 0.0f;  
  float density2 = 0.0f;
  float density3 = 0.0f;

  float data_x;
  float data_y;
  float loc_x ;
  float loc_y ;
  float loc_z ;
  float loc_w ;

  for (int z = zL; z <= zH; z++){
    for (int y = yL; y <= yH; y++){
      __global const unsigned int *addr = binStartAddr_g+z*SIZE_XY_VAL+ y*GRIDSIZE_VAL1;
      const unsigned int start = *(addr+xL);
      const unsigned int end   = *(addr+xH+1);
      const unsigned int delta = end-start;
      for (int x = 0; x < ((delta+TILE-1)>>LOG_TILE); x++){
        int tileSize = ((delta-(x<<LOG_TILE)) > TILE) ? TILE : (delta-(x<<LOG_TILE));
        int globalIdx = flatIdx + (x<<LOG_TILE);
        barrier(CLK_LOCAL_MEM_FENCE );
        if(flatIdx < tileSize){
          /*float*/ data_x = dataptr_g[(start+globalIdx) * 2];
          /*float*/ data_y = dataptr_g[(start+globalIdx) * 2 + 1];
          /*float*/ loc_x = ((__global float*)(dataptr_g+2*f2_offset))[4*(start+globalIdx)  ];
          /*float*/ loc_y = ((__global float*)(dataptr_g+2*f2_offset))[4*(start+globalIdx)+1];
          /*float*/ loc_z = ((__global float*)(dataptr_g+2*f2_offset))[4*(start+globalIdx)+2];
          /*float*/ loc_w = ((__global float*)(dataptr_g+2*f2_offset))[4*(start+globalIdx)+3];

          real_s[flatIdx] = data_x;
          imag_s[flatIdx] = data_y;
          kx_s  [flatIdx] = loc_x;
          ky_s  [flatIdx] = loc_y;
          kz_s  [flatIdx] = loc_z;
          sdc_s [flatIdx] = loc_w;
        }
        barrier(CLK_LOCAL_MEM_FENCE );

        // promoted due to CEAN bug
        float v, v0, v1, v2, v3, w;
        w = flatIdx;

        for (int j=0; j< tileSize; j++){
          const float real = real_s[j];
          const float imag = imag_s[j];
          const float sdc = sdc_s[j];

          if((real != 0.0f || imag != 0.0f) && sdc != 0.0f){
            v0 = (kx_s[j]-X)*(kx_s[j]-X);
            v0 += (ky_s[j]-Y)*(ky_s[j]-Y);

            /*const float*/ v = v0 + (kz_s[j]-Z)*(kz_s[j]-Z);
            if(v<CUTOFF2_VAL){
              /*const float*/ w = kernel_value(beta*sqrt(1.0f-(v*ONE_OVER_CUTOFF2_VAL))) *sdc;
              pt_x += w*real;
              pt_y += w*imag;
              density += 1.0f;
            }

            /*const float*/ v1 = v0 + (kz_s[j]-Z1)*(kz_s[j]-Z1);
            if(v1<CUTOFF2_VAL){
              /*const float*/ w = kernel_value(beta*sqrt(1.0f-(v1*ONE_OVER_CUTOFF2_VAL))) *sdc;
              pt1_x += w*real;
              pt1_y += w*imag;
              density1 += 1.0f;
            }

            /*const float*/ v2 = v0 + (kz_s[j]-Z2)*(kz_s[j]-Z2);
            if(v2<CUTOFF2_VAL){
              /*const float*/ w = kernel_value(beta*sqrt(1.0f-(v2*ONE_OVER_CUTOFF2_VAL))) *sdc;
              pt2_x += w*real;
              pt2_y += w*imag;
              density2 += 1.0f;
            }

            /*const float*/ v3 = v0 + (kz_s[j]-Z3)*(kz_s[j]-Z3);
            if(v3<CUTOFF2_VAL){
              /*const float*/ w = kernel_value(beta*sqrt(1.0f-(v3*ONE_OVER_CUTOFF2_VAL))) *sdc;
              pt3_x += w*real;
              pt3_y += w*imag;
              density3 += 1.0f;
            }
          }
        }
      }
    }
  }

  gridData_g[2*idx] = pt_x;
  gridData_g[2*idx+1] = pt_y;
  sampleDensity_g[idx+0] = density;

  gridData_g[2*idx1] = pt1_x;
  gridData_g[2*idx1+1] = pt1_y;
  sampleDensity_g[idx1+0] = density1;

  gridData_g[2*idx2] = pt2_x;
  gridData_g[2*idx2+1] = pt2_y;
  sampleDensity_g[idx2+0] = density2;

  gridData_g[2*idx3] = pt3_x;
  gridData_g[2*idx3+1] = pt3_y;
  sampleDensity_g[idx3+0] = density3;
}

