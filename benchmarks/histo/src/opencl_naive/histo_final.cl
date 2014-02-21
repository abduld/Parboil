/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

/* Combine all the sub-histogram results into one final histogram */
__kernel void histo_final_kernel (
    unsigned int histo_height, 
    unsigned int histo_width,
    __global uint *global_subhisto,
    __global uchar *final_histo) //final output
{
	  uint tid = get_global_id(0);

	  if (tid < (histo_height*histo_width))
    {
    	  unsigned int internal_histo_data = global_subhisto[tid];
    	  uchar final_histo_data = min(internal_histo_data, 255U);
    	  final_histo[tid] = final_histo_data;
    }
}
