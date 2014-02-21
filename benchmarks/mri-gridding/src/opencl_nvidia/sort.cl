/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/
 
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable 

#ifndef CUTOFF2_VAL
#define CUTOFF2_VAL 6.250000
#define CUTOFF_VAL 2.500000
#define CEIL_CUTOFF_VAL 3.000000
#define GRIDSIZE_VAL1 256
#define GRIDSIZE_VAL2 256
#define GRIDSIZE_VAL3 256
#define SIZE_XY_VAL 65536
#define ONE_OVER_CUTOFF2_VAL 0.160000
#endif

#ifndef DYN_LOCAL_MEM_SIZE
#define DYN_LOCAL_MEM_SIZE 1092
#endif

#define UINT32_MAX (4294967295/4)
#define BITS 4
#define LNB 4

#define SORT_BS 256

// #define CONFLICT_FREE_OFFSET(index) ((index) >> LNB + (index) >> (2*LNB))
// #define CONFLICT_FREE_OFFSET(index) (((unsigned int)(index) >> min((unsigned int)(LNB)+(index), (unsigned int)(32-(2*LNB))))>>(2*LNB))
#define CONFLICT_FREE_OFFSET(index)     (((unsigned int)(index) >> (unsigned int)(32-(2*LNB)))>>(2*LNB))
#define BLOCK_P_OFFSET (4*SORT_BS+1+(4*SORT_BS+1)/16+(4*SORT_BS+1)/64)

__kernel void splitSort(int numElems, int iter, 
                                 __global unsigned int* keys, 
                                 __global unsigned int* values, 
                                 __global unsigned int* histo)
{
    __local unsigned int flags[BLOCK_P_OFFSET];
    __local unsigned int histo_s[1<<BITS];

    const unsigned int tid = get_local_id(0);
    const unsigned int gid = get_group_id(0)*4*SORT_BS+4*get_local_id(0);

    // Copy input to shared mem. Assumes input is always even numbered
    uint lkey_x = UINT32_MAX;
    uint lkey_y = UINT32_MAX;
    uint lkey_z = UINT32_MAX;
    uint lkey_w = UINT32_MAX;
    uint lvalue_x = 0;
    uint lvalue_y = 0;
    uint lvalue_z = 0;
    uint lvalue_w = 0;
    if (gid < numElems){
      lkey_x = *(keys+gid);
      lkey_y = *(keys+gid+1);
      lkey_z = *(keys+gid+2);
      lkey_w = *(keys+gid+3);
      lvalue_x = *(values+gid);
      lvalue_y = *(values+gid+1);
      lvalue_z = *(values+gid+2);
      lvalue_w = *(values+gid+3);
    }

    if(tid < (1<<BITS)){
      histo_s[tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

    atom_add(histo_s+((lkey_x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atom_add(histo_s+((lkey_y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atom_add(histo_s+((lkey_z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atom_add(histo_s+((lkey_w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);

    int _x = 4*tid;
    int _y = 4*tid+1;
    int _z = 4*tid+2;
    int _w = 4*tid+3;

    unsigned int lsz0 = get_local_size(0);
    for (int i=BITS*iter; i<BITS*(iter+1);i++){
      const uint flag_x = (lkey_x>>i)&0x1;
      const uint flag_y = (lkey_y>>i)&0x1;
      const uint flag_z = (lkey_z>>i)&0x1;
      const uint flag_w = (lkey_w>>i)&0x1;

      flags[_x+CONFLICT_FREE_OFFSET(_x)] = 1<<(16*flag_x);
      flags[_y+CONFLICT_FREE_OFFSET(_y)] = 1<<(16*flag_y);
      flags[_z+CONFLICT_FREE_OFFSET(_z)] = 1<<(16*flag_z);
      flags[_w+CONFLICT_FREE_OFFSET(_w)] = 1<<(16*flag_w);

      /* promoted due to CEAN bug */
      unsigned int _i;
      unsigned int ai;
      unsigned int bi;
      unsigned int t;

      // scan (flags);
      {
        __local unsigned int* s_data = flags;
        unsigned int thid = tid;
      
        barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();
      
        s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)] += s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)];
        s_data[2*(lsz0+thid)+1+CONFLICT_FREE_OFFSET(2*(lsz0+thid)+1)] += s_data[2*(lsz0+thid)+CONFLICT_FREE_OFFSET(2*(lsz0+thid))];
      
        unsigned int stride = 2;
        for (unsigned int d = lsz0; d > 0; d >>= 1)
        {
          barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();
      
          if (thid < d)
          {
            /*unsigned int*/ _i  = 2*stride*thid;
            /*unsigned int*/ ai = _i + stride - 1;
            /*unsigned int*/ bi = ai + stride;
      
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
      
            s_data[bi] += s_data[ai];
          }
      
          stride *= 2;
        }
      
        if (thid == 0){
#if 0
          unsigned int last = 4*lsz0-1;
          last += CONFLICT_FREE_OFFSET(last);
          s_data[4*lsz0+CONFLICT_FREE_OFFSET(4*lsz0)] = s_data[last];
          s_data[last] = 0;
#else
          #define LAST ((4*lsz0-1) + CONFLICT_FREE_OFFSET(((4*lsz0-1))))
          s_data[4*lsz0+CONFLICT_FREE_OFFSET(4*lsz0)] = s_data[LAST];
          s_data[LAST] = 0;
#endif
        }

        for (unsigned int d = 1; d <= lsz0; d *= 2)
        {
          stride >>= 1;
      
          barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();
      
          if (thid < d)
          {
            /*unsigned int*/ _i  = 2*stride*thid;
            /*unsigned int*/ ai = _i + stride - 1;
            /*unsigned int*/ bi = ai + stride;
      
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
      
            /* unsigned int*/ t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();
      
        unsigned int temp = s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)];
        s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)] = s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)];
        s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)] += temp;
      
        unsigned int temp2 = s_data[2*(lsz0+thid)+CONFLICT_FREE_OFFSET(2*(lsz0+thid))];
        s_data[2*(lsz0+thid)+CONFLICT_FREE_OFFSET(2*(lsz0+thid))] = s_data[2*(lsz0+thid)+1+CONFLICT_FREE_OFFSET(2*(lsz0+thid)+1)];
        s_data[2*(lsz0+thid)+1+CONFLICT_FREE_OFFSET(2*(lsz0+thid)+1)] += temp2;
      
        barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();
      }

      _x = (flags[_x+CONFLICT_FREE_OFFSET(_x)]>>(16*flag_x))&0xFFFF;
      _y = (flags[_y+CONFLICT_FREE_OFFSET(_y)]>>(16*flag_y))&0xFFFF;
      _z = (flags[_z+CONFLICT_FREE_OFFSET(_z)]>>(16*flag_z))&0xFFFF;
      _w = (flags[_w+CONFLICT_FREE_OFFSET(_w)]>>(16*flag_w))&0xFFFF;

      unsigned short offset = flags[4*lsz0+CONFLICT_FREE_OFFSET(4*lsz0)]&0xFFFF;
      _x += (flag_x) ? offset : 0;
      _y += (flag_y) ? offset : 0;
      _z += (flag_z) ? offset : 0;
      _w += (flag_w) ? offset : 0;

      barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();
    }

    // Write result.
    if (gid < numElems){
      keys[get_group_id(0)*4*SORT_BS+_x] = lkey_x;
      keys[get_group_id(0)*4*SORT_BS+_y] = lkey_y;
      keys[get_group_id(0)*4*SORT_BS+_z] = lkey_z;
      keys[get_group_id(0)*4*SORT_BS+_w] = lkey_w;

      values[get_group_id(0)*4*SORT_BS+_x] = lvalue_x;
      values[get_group_id(0)*4*SORT_BS+_y] = lvalue_y;
      values[get_group_id(0)*4*SORT_BS+_z] = lvalue_z;
      values[get_group_id(0)*4*SORT_BS+_w] = lvalue_w;
    }
    if (tid < (1<<BITS)){
      histo[get_num_groups(0)*tid+get_group_id(0)] = histo_s[tid];
    }
}

__kernel void splitRearrange (int numElems, int iter, 
                                __global unsigned int* keys_i, 
                                __global unsigned int* keys_o, 
                                __global unsigned int* values_i, 
                                __global unsigned int* values_o, 
                                __global unsigned int* histo){
  __local unsigned int histo_s[(1<<BITS)];
  __local uint array_s[4*SORT_BS];
  const unsigned int tid = get_local_id(0);
  int index = get_group_id(0)*4*SORT_BS + 4*get_local_id(0);

  if (tid < (1<<BITS)){
    histo_s[tid] = histo[get_num_groups(0)*tid+get_group_id(0)];
  }


  uint mine_x;
  uint mine_y;
  uint mine_z;
  uint mine_w;
  uint value_x = 0;
  uint value_y = 0;
  uint value_z = 0;
  uint value_w = 0;
  if (index < numElems){
    mine_x = *(__global uint*)(keys_i+index);
    mine_y = *(__global uint*)(keys_i+index+1);
    mine_z = *(__global uint*)(keys_i+index+2);
    mine_w = *(__global uint*)(keys_i+index+3);
    value_x = *(__global uint*)(values_i+index);
    value_y = *(__global uint*)(values_i+index+1);
    value_z = *(__global uint*)(values_i+index+2);
    value_w = *(__global uint*)(values_i+index+3);
  } else {
    mine_x = UINT32_MAX;
    mine_y = UINT32_MAX;
    mine_z = UINT32_MAX;
    mine_w = UINT32_MAX;
  }

#if 0
  uint4 masks = (uint4) (
                 (mine_x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine_y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine_z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine_w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter) );
#else
  uint masks_x = (mine_x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter);
  uint masks_y = (mine_y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter);
  uint masks_z = (mine_z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter);
  uint masks_w = (mine_w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter);
#endif

#if 0
//  ((__local uint4*)array_s)[get_local_id(0)] = masks;
  vstore4(masks, get_local_id(0), (__local uint *)array_s);
#else

#if 0
  ((__local uint4*)array_s)[get_local_id(0)] = masks;
#else
  ((__local uint*) array_s)[4*get_local_id(0)  ] = masks_x;
  ((__local uint*) array_s)[4*get_local_id(0)+1] = masks_y;
  ((__local uint*) array_s)[4*get_local_id(0)+2] = masks_z;
  ((__local uint*) array_s)[4*get_local_id(0)+3] = masks_w;
#endif

#endif
  
  barrier(CLK_LOCAL_MEM_FENCE ); //__syncthreads();

#if 0
  uint4 new_index = (uint4) ( histo_s[masks.x],histo_s[masks.y],histo_s[masks.z],histo_s[masks.w] );
#else
  uint new_index_x = histo_s[masks_x];
  uint new_index_y = histo_s[masks_y];
  uint new_index_z = histo_s[masks_z];
  uint new_index_w = histo_s[masks_w];
#endif

  int i = 4*get_local_id(0)-1;
  
  while (i >= 0){
    if (array_s[i] == masks_x){
      new_index_x = new_index_x+1;
      i--;
    } else {
      break;
    }
  }

  new_index_y = (masks_y == masks_x) ? new_index_x+1 : new_index_y;
  new_index_z = (masks_z == masks_y) ? new_index_y+1 : new_index_z;
  new_index_w = (masks_w == masks_z) ? new_index_z+1 : new_index_w;

  if (index < numElems){
    keys_o[new_index_x] = mine_x;
    values_o[new_index_x] = value_x;

    keys_o[new_index_y] = mine_y;
    values_o[new_index_y] = value_y;

    keys_o[new_index_z] = mine_z;
    values_o[new_index_z] = value_z;

    keys_o[new_index_w] = mine_w;
    values_o[new_index_w] = value_w; 
  }  
}

