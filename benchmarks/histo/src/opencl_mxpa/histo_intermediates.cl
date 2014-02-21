/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

void calculateBin (
        __const unsigned int bin,
        __global uint *global_subhisto)
{
	      uint old_val = global_subhisto[bin];
	      if (old_val < 255)
	      	atom_inc(&global_subhisto[bin]);
}

__kernel void histo_intermediates_kernel (
        __global uint *input,
        unsigned int input_pitch,
        __global uint *global_subhisto)
{
        unsigned int idx = 0;
        unsigned int bin_value;
        __global uint * __local load_bin;
        load_bin = input + (input_pitch*get_group_id(0));

        while (idx < input_pitch) {
        	unsigned int my_idx = idx + get_local_id(0);
        	if (my_idx < input_pitch) {
            bin_value = load_bin[my_idx];
            calculateBin (bin_value, global_subhisto);
          }
          idx += get_local_size(0);
        }
}
