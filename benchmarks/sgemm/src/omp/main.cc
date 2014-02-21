// (C) Copyright 2013, University of Illinois. All Rights Reserved.
// Xuhao Chen <cxh.nudt@gmail.com>
// Main entry of dense matrix-matrix multiplication kernel
// OpenMP implementation

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <parboil.h>
#include <iostream>

extern void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc );
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

int main (int argc, char *argv[]) {
  struct pb_Parameters *params;
  struct pb_TimerSet timers;
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;
  pb_InitializeTimerSet(&timers);
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL)) {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
  }
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  readColMajorMatrixFile(params->inpFiles[0], matArow, matAcol, matA);
  readColMajorMatrixFile(params->inpFiles[2], matBcol, matBrow, matBT);
  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  std::vector<float> matC(matArow*matBcol);
  basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f,
      &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);
  if (params->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    writeColMajorMatrixFile(params->outFile, matArow, matBcol, matC); 
  }
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  double CPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_COMPUTE]));
  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/CPUtime/1e9 << std::endl;
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  return 0;
}
