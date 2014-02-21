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
	  unsigned int internal_histo_data;
	  uchar final_histo_data;

	  if (tid < (histo_height*histo_width))
    {
    	  internal_histo_data = global_subhisto[tid];
    	  final_histo_data = min (internal_histo_data, 255);
    	  final_histo[tid] = final_histo_data;
    }
}
