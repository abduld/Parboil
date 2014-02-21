/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

__kernel void spmv_jds_naive(__global float *dst_vector, __global float *d_data,
		       	     __global int *d_index, __global int *d_perm,
		             __global float *x_vec, const int _dim, 
		             __constant int *jds_ptr_int,
		             __constant int *sh_zcnt_int)
{
  	int ix = get_global_id(0);
    float sum = 0.0f;
    float d, t;
    int j, in;
    __local int bound, k;

  	if (ix < _dim) {
    		bound=sh_zcnt_int[get_group_id(0)];
	    	for(k=0;k<bound;k++)
    		{	  
      			j = jds_ptr_int[k] + ix;    
      			in = d_index[j]; 
  
      			d = d_data[j];
      			t = x_vec[in];

      			sum += d*t; 
    		}  
  
    		dst_vector[d_perm[ix]] = sum; 
  	}
}
