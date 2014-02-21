/***************************************************************************
 *cr
 *cr            (C) Copyright 2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "file.h"

void readImage(const char *fName, unsigned int *hh_DataA, unsigned DATA_SIZE)
{

  FILE *File;
  unsigned short temp;
  int y;

  if((File = fopen(fName, "rb")) != NULL) {
    for (y=0; y < DATA_SIZE; y++){
      int fr = fread(&temp, sizeof(unsigned short), 1, File);
      hh_DataA[y] = (unsigned int)ByteSwap16(temp);
      if(hh_DataA[y] >= 4096) hh_DataA[y] = 4095;
    }
    fclose(File);
  } else {
    printf("%s does not exist\n", fName);
    exit(1);
  }


}


void writeVector(const char *fName, unsigned int *vec_h, unsigned size)
{
    FILE* fp = fopen(fName, "w");
    if (fp == NULL) FATAL("Cannot open output file");
    fwrite(&size, sizeof(unsigned), 1, fp);
    fwrite(vec_h, sizeof(unsigned int), size, fp);
    fclose(fp);
}
