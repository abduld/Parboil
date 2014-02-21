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

void _calculateBin (
        __const unsigned int bin,
        __global uchar *sm_mapping)
{
        unsigned char offset  =  bin        %   4;
        unsigned char indexlo = (bin >>  2) % 256;
        unsigned char indexhi = (bin >> 10) %  KB;
        unsigned char block   =  bin / BINS_PER_BLOCK;

        offset *= 8;

        *(sm_mapping) = block;
        *(sm_mapping+1) = indexhi;
        *(sm_mapping+2) = indexlo;
        *(sm_mapping+3) = offset;
}

__kernel void calculateBin (
        __const unsigned int bin,
        __global uchar *sm_mapping)
{
        unsigned char offset  =  bin        %   4;
        unsigned char indexlo = (bin >>  2) % 256;
        unsigned char indexhi = (bin >> 10) %  KB;
        unsigned char block   =  bin / BINS_PER_BLOCK;

        offset *= 8;

        *(sm_mapping) = block;
        *(sm_mapping+1) = indexhi;
        *(sm_mapping+2) = indexlo;
        *(sm_mapping+3) = offset;
}

__kernel void histo_intermediates_kernel (
        __global uint *input,
        unsigned int height,
        unsigned int width,
        unsigned int input_pitch,
        __global uchar *sm_mappings)
{
        int threadIdxx = get_local_id(0);
        int blockDimx = get_local_size(0);
        unsigned int line = UNROLL * (get_group_id(0));// 16 is the unroll factor;

        __global uint *load_bin = input + 2 * (line * input_pitch + threadIdxx);

        unsigned int store = line * width + threadIdxx;
        bool skip = (width % 2) && (threadIdxx == (blockDimx - 1));

        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
                // uint2 bin_value = *load_bin;
                uint bin_value_x = *load_bin;
                uint bin_value_y = *(load_bin+1);

                _calculateBin (
                        bin_value_x,
                        &sm_mappings[4*store]
                );

                if (!skip) { _calculateBin (
                        bin_value_y,
                        &sm_mappings[4*(store + blockDimx)]
                );
                }

                load_bin += 2*input_pitch;
                store += width;
        }
}

__kernel void histo_intermediates_kernel_compat (
        __global uint2 *input,
        unsigned int height,
        unsigned int width,
        unsigned int input_pitch,
        __global uchar *sm_mappings)
{
        int threadIdxx = get_local_id(0);
        int blockDimx = input_pitch; //get_local_size(0);
        
        int tid2 = get_local_id(0) + get_local_size(0);
        
        unsigned int line = UNROLL * (get_group_id(0));// 16 is the unroll factor;

        __global uint *load_bin = input + 2 * (line * input_pitch + threadIdxx);
        __global uint *load_bin2 = input + 2 * (line * input_pitch + tid2);

        unsigned int store = line * width + threadIdxx;
        unsigned int store2 = line * width + tid2;
        
        bool skip = (width % 2) && (threadIdxx == (input_pitch - 1));
        bool skip2 = (width % 2) && (tid2 == (input_pitch - 1));

        bool does2 = tid2 < input_pitch;

        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
                uint bin_value_x = *load_bin;
                uint bin_value_y = *(load_bin+1);


                _calculateBin (
                        bin_value_x,
                        &sm_mappings[4*store]
                );

                if (!skip) { _calculateBin (
                        bin_value_y,
                        &sm_mappings[4*(store + blockDimx)]
                );
                }

                load_bin += 2*input_pitch;
                store += width;
                
                if (does2) {
                  uint bin_val2_x = *load_bin2;
                  uint bin_val2_y = *(load_bin2+1);
                                
                  _calculateBin (
                        bin_val2_x,
                        &sm_mappings[store2]
                  );

                  if (!skip) {    _calculateBin (
                        bin_val2_y,
                        &sm_mappings[store2 + blockDimx]
                  );
                  }

                  load_bin2 += 2*input_pitch;
                  store2 += width;
                }
        }
}


