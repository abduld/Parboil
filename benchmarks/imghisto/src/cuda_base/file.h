/***************************************************************************
 *cr
 *cr            (C) Copyright 2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef __FILEH__
#define __FILEH__

#ifdef __cplusplus
extern "C" {
#endif

#define ByteSwap16(n) ( ((((unsigned int) n) << 8) & 0xFF00) | ((((unsigned int) n) >> 8) & 0x00FF) )

void readImage(const char *fName, unsigned int *hh_DataA, unsigned DATA_SIZE);
void writeVector(const char* fName, unsigned int *vec_h, unsigned size);

#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
