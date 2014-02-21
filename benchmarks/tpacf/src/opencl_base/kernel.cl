/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include "model.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

  __kernel
void gen_hists(__global hist_t* histograms, __global float* all_x_data,
    __constant float* dev_binb, int NUM_SETS, int NUM_ELEMENTS)
{
  __global float* all_y_data = all_x_data + NUM_ELEMENTS*(NUM_SETS+1); 
  __global float* all_z_data = all_y_data + NUM_ELEMENTS*(NUM_SETS+1);

  unsigned int bx = get_group_id(0);
  unsigned int tid = get_local_id(0);
  bool do_self = (bx < (NUM_SETS + 1));

  __global hist_t* block_histogram = histograms + NUM_BINS * bx;
  __global float* data_x;
  __global float* data_y;
  __global float* data_z;
  __global float* random_x;
  __global float* random_y;
  __global float* random_z;

  float distance;
  float random_x_s;
  float random_y_s;
  float random_z_s;

  unsigned int bin_index;
  // XXX: HSK: Bad trick to walkaround the compiler bug
  unsigned int min = get_local_id(0); // 0
  unsigned int max = get_local_id(0); // NUM_BINS

  if(tid < NUM_BINS)
  {
    block_histogram[tid] = 0;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  // Get pointers set up
  if( !do_self)
  {
    data_x = all_x_data;
    data_y = all_y_data;
    data_z = all_z_data;

    random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
    random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
    random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
  }
  else
  {
    random_x = all_x_data + NUM_ELEMENTS * (bx);
    random_y = all_y_data + NUM_ELEMENTS * (bx);
    random_z = all_z_data + NUM_ELEMENTS * (bx);

    data_x = random_x;
    data_y = random_y;
    data_z = random_z;
  }

  // Iterate over all random points
  for(unsigned int j = 0; j < NUM_ELEMENTS; j += BLOCK_SIZE)
  {
    if(tid + j < NUM_ELEMENTS)
    {
      random_x_s = random_x[tid + j];
      random_y_s = random_y[tid + j];
      random_z_s = random_z[tid + j];
    }

    // Iterate over all data points
    // If do_self, then use a tighter bound on the number of data points.
    for(unsigned int k = 0;
        k < NUM_ELEMENTS && (do_self ? k < j + BLOCK_SIZE : 1); k++)
    {
      // do actual calculations on the values:
      distance = data_x[k] * random_x_s + 
        data_y[k] * random_y_s + 
        data_z[k] * random_z_s ;

      // run binary search to find bin_index
#if 0 /* XXX: HSK: Bad trick to walkaround the compiler bug */
      min = 0;
      max = NUM_BINS;
#else
      if (get_local_id(0) >= 0) {
        min = 0;
        max = NUM_BINS;
      }
#endif
      {
        unsigned int k2;

        while (max > min+1)
        {
          k2 = (min + max) / 2;
          // k2 = (min + max) >> 1;
          if (distance >= dev_binb[k2]) 
            max = k2;
          else 
            min = k2;
        }
        bin_index = max - 1;
      }

      if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
          (!do_self || (tid + j > k)) && ((tid + j) < NUM_ELEMENTS))
      {
        atom_inc(&(block_histogram[bin_index]));
      }
    }
  }
}

// **===-----------------------------------------------------------===**

#endif // _PRESCAN_CU_
