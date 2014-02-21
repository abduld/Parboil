/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#ifndef PRESCAN_THREADS
#define PRESCAN_THREADS   512
#define KB                24
#define BLOCK_X           14
#define UNROLL            16
#define BINS_PER_BLOCK    (KB * 1024)
#endif

/* Combine all the sub-histogram results into one final histogram */
__kernel void histo_final_kernel (
    unsigned int sm_range_min, 
    unsigned int sm_range_max,
    unsigned int histo_height, 
    unsigned int histo_width,
    __global unsigned int *global_subhisto,
    __global unsigned int *global_histo,
    __global unsigned int *global_overflow,
    __global unsigned int *final_histo) //final output
{
    unsigned int blockDimx = get_local_size(0);
    unsigned int gridDimx = get_num_groups(0);
    unsigned int start_offset = get_local_id(0) + get_group_id(0) * blockDimx;
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;

    /* Clear lower region of global histogram */
    for (unsigned int i = start_offset; i < size_low_histo/4; i += gridDimx * blockDimx)
    {
        ushort global_histo_data_x = ((__global ushort*)global_histo)[4*i];
        ushort global_histo_data_y = ((__global ushort*)global_histo)[4*i+1];
        ushort global_histo_data_z = ((__global ushort*)global_histo)[4*i+2];
        ushort global_histo_data_w = ((__global ushort*)global_histo)[4*i+3];

        ((__global ushort*)global_histo)[4*i] = 0;
        ((__global ushort*)global_histo)[4*i+1] = 0;
        ((__global ushort*)global_histo)[4*i+2] = 0;
        ((__global ushort*)global_histo)[4*i+3] = 0;

        global_histo_data_x = min (global_histo_data_x, (ushort) 255);
        global_histo_data_y = min (global_histo_data_y, (ushort) 255);
        global_histo_data_z = min (global_histo_data_z, (ushort) 255);
        global_histo_data_w = min (global_histo_data_w, (ushort) 255);

        ((__global uchar*)final_histo)[4*i] = (uchar)global_histo_data_x;
        ((__global uchar*)final_histo)[4*i+1] = (uchar)global_histo_data_y;
        ((__global uchar*)final_histo)[4*i+2] = (uchar)global_histo_data_z;
        ((__global uchar*)final_histo)[4*i+3] = (uchar)global_histo_data_w;
    }

    /* Clear the middle region of the overflow buffer */
    for (unsigned int i = (size_low_histo/4) + start_offset; i < (size_low_histo+size_mid_histo)/4; i += gridDimx * blockDimx)
    {
        uint global_histo_data_x = ((__global uint*)global_overflow)[4*i  ];
        uint global_histo_data_y = ((__global uint*)global_overflow)[4*i+1];
        uint global_histo_data_z = ((__global uint*)global_overflow)[4*i+2];
        uint global_histo_data_w = ((__global uint*)global_overflow)[4*i+3];

        ((__global uint*)global_overflow)[4*i  ] = 0;
        ((__global uint*)global_overflow)[4*i+1] = 0;
        ((__global uint*)global_overflow)[4*i+2] = 0;
        ((__global uint*)global_overflow)[4*i+3] = 0;

        unsigned int bin4in0 = ((__global unsigned int*)global_subhisto)[i*4];
        unsigned int bin4in1 = ((__global unsigned int*)global_subhisto)[i*4+1];
        unsigned int bin4in2 = ((__global unsigned int*)global_subhisto)[i*4+2];
        unsigned int bin4in3 = ((__global unsigned int*)global_subhisto)[i*4+3];

        uint internal_histo_data_x = min (bin4in0, (unsigned int) 255);
        uint internal_histo_data_y = min (bin4in1, (unsigned int) 255);
        uint internal_histo_data_z = min (bin4in2, (unsigned int) 255);
        uint internal_histo_data_w = min (bin4in3, (unsigned int) 255);

        ((__global uchar*)final_histo)[4*i  ] = (uchar) internal_histo_data_x;
        ((__global uchar*)final_histo)[4*i+1] = (uchar) internal_histo_data_y;
        ((__global uchar*)final_histo)[4*i+2] = (uchar) internal_histo_data_z;
        ((__global uchar*)final_histo)[4*i+3] = (uchar) internal_histo_data_w;
    }

    /* Clear the upper region of global histogram */
    for (unsigned int i = ((size_low_histo+size_mid_histo)/4) + start_offset; i < (histo_height*histo_width)/4; i += gridDimx * blockDimx)
    {
        ushort global_histo_data_x = ((__global ushort*)global_histo)[4*i  ];
        ushort global_histo_data_y = ((__global ushort*)global_histo)[4*i+1];
        ushort global_histo_data_z = ((__global ushort*)global_histo)[4*i+2];
        ushort global_histo_data_w = ((__global ushort*)global_histo)[4*i+3];

        ((__global ushort*)global_histo)[4*i] = 0;
        ((__global ushort*)global_histo)[4*i+1] = 0;
        ((__global ushort*)global_histo)[4*i+2] = 0;
        ((__global ushort*)global_histo)[4*i+3] = 0;

        global_histo_data_x = min (global_histo_data_x, (ushort) 255);
        global_histo_data_y = min (global_histo_data_y, (ushort) 255);
        global_histo_data_z = min (global_histo_data_z, (ushort) 255);
        global_histo_data_w = min (global_histo_data_w, (ushort) 255);

        ((__global uchar*)final_histo)[4*i  ] = (uchar) global_histo_data_x;
        ((__global uchar*)final_histo)[4*i+1] = (uchar) global_histo_data_y;
        ((__global uchar*)final_histo)[4*i+2] = (uchar) global_histo_data_z;
        ((__global uchar*)final_histo)[4*i+3] = (uchar) global_histo_data_w;
    }
}
