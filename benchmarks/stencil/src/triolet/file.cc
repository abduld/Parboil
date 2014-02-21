/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <inttypes.h>
#include <TrioletData.h>

#include "file.h"

using namespace Triolet;

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif


void outputData(char* fName, Array3<Float> A,int nx,int ny,int nz)
{
  FILE* fid = fopen(fName, "w");
  uint32_t tmp32;
  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }
  tmp32 = nx*ny*nz;
  fwrite(&tmp32, sizeof(uint32_t), 1, fid);
  int i,j,k;
  for (i = 0; i < nz; i++) {
    for (j = 0; j < ny; j++) {
      for (k = 0; k < nx; k++) {
        float f = A.at(i,j,k);
        fwrite(&f, sizeof(float), 1, fid);
      }
    }
  }

  fclose (fid);
}
