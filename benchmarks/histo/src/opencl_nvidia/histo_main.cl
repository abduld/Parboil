/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#ifndef PRESCAN_THREADS
#define PRESCAN_THREADS   512
#define KB                48
#define BLOCK_X           14
#define UNROLL            16
#define BINS_PER_BLOCK    (KB * 1024)
#endif

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

void testIncrementGlobal (
        __global unsigned int *global_histo,
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        const uchar sm_x,
        const uchar sm_y,
        const uchar sm_z,
        const uchar sm_w)
{
        const unsigned int range = sm_x;
        const unsigned int indexhi = sm_y;
        const unsigned int indexlo = sm_z;
        const unsigned int offset  = sm_w;

        /* Scan for inputs that are outside the central region of histogram */
        if (range < sm_range_min || range > sm_range_max)
        {
                const unsigned int bin = range * BINS_PER_BLOCK + offset / 8 + (indexlo << 2) + (indexhi << 10);
                const unsigned int bin_div2 = bin / 2;
                const unsigned int bin_offset = (bin % 2 == 1) ? 16 : 0;

                unsigned int old_val = global_histo[bin_div2];
                unsigned short old_bin = (old_val >> bin_offset) & 0xFFFF;

                if (old_bin < 255)
                {
                        atom_add (&global_histo[bin_div2], 1 << bin_offset);
                }
        }
}

void testIncrementLocal (
        __global unsigned int *global_overflow,
        __local unsigned int* smem, // __local unsigned int smem[KB][256],
        const unsigned int myRange,
        const uchar sm_x,
        const uchar sm_y,
        const uchar sm_z,
        const uchar sm_w)
{
        const unsigned int range = sm_x;
        const unsigned int indexhi = sm_y;
        const unsigned int indexlo = sm_z;
        const unsigned int offset  = sm_w;

        // Scan for inputs that are inside the central region of histogram 
        if (range == myRange)
        {
                /* Atomically increment shared memory */
                unsigned int add = (unsigned int)(1 << offset);
                unsigned int prev = atom_add (&smem[indexhi * 256 + indexlo], add);

                /* Check if current bin overflowed */
                unsigned int prev_bin_val = (prev >> offset) & 0x000000FF;

                /* If there was an overflow, record it and record if it cascaded into other bins */
                if (prev_bin_val == 0x000000FF)
                {
                        const unsigned int bin =
                                range * BINS_PER_BLOCK +
                                offset / 8 + (indexlo << 2) + (indexhi << 10);

                        bool can_overflow_to_bin_plus_1 = (offset < 24) ? true : false;
                        bool can_overflow_to_bin_plus_2 = (offset < 16) ? true : false;
                        bool can_overflow_to_bin_plus_3 = (offset <  8) ? true : false;

                        bool overflow_into_bin_plus_1 = false;
                        bool overflow_into_bin_plus_2 = false;
                        bool overflow_into_bin_plus_3 = false;

                        unsigned int prev_bin_plus_1_val = (prev >> (offset +  8)) & 0x000000FF;
                        unsigned int prev_bin_plus_2_val = (prev >> (offset + 16)) & 0x000000FF;
                        unsigned int prev_bin_plus_3_val = (prev >> (offset + 24)) & 0x000000FF;

                        if (can_overflow_to_bin_plus_1 &&        prev_bin_val == 0x000000FF) overflow_into_bin_plus_1 = true;
                        if (can_overflow_to_bin_plus_2 && prev_bin_plus_1_val == 0x000000FF) overflow_into_bin_plus_2 = true;
                        if (can_overflow_to_bin_plus_3 && prev_bin_plus_2_val == 0x000000FF) overflow_into_bin_plus_3 = true;

                        unsigned int bin_plus_1_add;
                        unsigned int bin_plus_2_add;
                        unsigned int bin_plus_3_add;

                        if (overflow_into_bin_plus_1) bin_plus_1_add = (prev_bin_plus_1_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
                        if (overflow_into_bin_plus_2) bin_plus_2_add = (prev_bin_plus_2_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
                        if (overflow_into_bin_plus_3) bin_plus_3_add = (prev_bin_plus_3_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;

                                                      atom_add (&global_overflow[bin],   256);
                        if (overflow_into_bin_plus_1) atom_add (&global_overflow[bin+1], bin_plus_1_add);
                        if (overflow_into_bin_plus_2) atom_add (&global_overflow[bin+2], bin_plus_2_add);
                        if (overflow_into_bin_plus_3) atom_add (&global_overflow[bin+3], bin_plus_3_add);
                }
        }
}

__kernel void histo_main_kernel (
        __global uchar *sm_mappings,
        unsigned int num_elements,
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        unsigned int histo_height,
        unsigned int histo_width,
        __global unsigned int *global_subhisto,
        __global unsigned int *global_histo,
        __global unsigned int *global_overflow)
{
        /* Most optimal solution uses 24 * 1024 bins per threadblock */
        __local unsigned int sub_histo[KB][256];

        /* Each threadblock contributes to a specific 24KB range of histogram,
         * and also scans every N-th line for interesting data.  N = gridDim.x
         */
        unsigned int blockDimx = get_local_size(0);
        unsigned int gridDimx = get_num_groups(0);
        unsigned int local_scan_range = sm_range_min + get_group_id(1);
        unsigned int local_scan_load = get_group_id(0) * blockDimx + get_local_id(0);

        // clearMemory (sub_histo);
        {
                __local unsigned int* smem = (__local unsigned int*) sub_histo;
                int i = get_local_id(0);
                int blockDimx = get_local_size(0);
                for (; i < BINS_PER_BLOCK / 4; i += blockDimx)
                {
                        ((__local unsigned int*)smem)[i] = 0;
                }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uchar sm_x;
        uchar sm_y;
        uchar sm_z;
        uchar sm_w;

        bool is_first_group;
        is_first_group = (get_group_id(1) == 0);

        if (is_first_group) // { get_group_id(1) == 0 )
        {
                /* Loop through and scan the input */
                while (local_scan_load < num_elements)
                {
                        // Read buffer 
                        sm_x = sm_mappings[4*local_scan_load];
                        sm_y = sm_mappings[4*local_scan_load+1];
                        sm_z = sm_mappings[4*local_scan_load+2];
                        sm_w = sm_mappings[4*local_scan_load+3];
                        local_scan_load += blockDimx * gridDimx;

                        /* Check input */
                        testIncrementLocal (
                                global_overflow,
                                (__local unsigned int *)sub_histo,
                                local_scan_range,
                                sm_x, sm_y, sm_z, sm_w
                        );
                        testIncrementGlobal (
                                global_histo,
                                sm_range_min,
                                sm_range_max,
                                sm_x, sm_y, sm_z, sm_w
                        );
                }
        }
        else
        {
                /* Loop through and scan the input */
                while (local_scan_load < num_elements)
                {
                        // Read buffer 
                        uchar sm_x = sm_mappings[4*local_scan_load];
                        uchar sm_y = sm_mappings[4*local_scan_load+1];
                        uchar sm_z = sm_mappings[4*local_scan_load+2];
                        uchar sm_w = sm_mappings[4*local_scan_load+3];
                        local_scan_load += blockDimx * gridDimx;

                        /* Check input */
                        testIncrementLocal (
                                global_overflow,
                                (__local unsigned int *)sub_histo,
                                local_scan_range,
                                sm_x, sm_y, sm_z, sm_w
                        );
                }
        }

        /* Store sub histogram to global memory */
        unsigned int store_index = get_group_id(0) * (histo_height * histo_width / 4) + (local_scan_range * BINS_PER_BLOCK / 4);//(local_scan_range * BINS_PER_BLOCK);

        barrier(CLK_LOCAL_MEM_FENCE);
        // copyMemory (&(global_subhisto[store_index]), sub_histo);
        {
                __global unsigned int* dst = &(global_subhisto[store_index]);
                __local unsigned int* src = (__local unsigned int*) sub_histo;
                int i = get_local_id(0);
                int blockDimx = get_local_size(0);
                for (; i < BINS_PER_BLOCK/4; i += blockDimx)
                {
                        dst[i] = ((__local unsigned int*)src)[i];
                }
        }
}
