// (C) Copyright 2013, University of Illinois, All Rights Reserved
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <parboil.h>
#include <iostream>
#include "sgemm_kernel.cc"

extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);


static void worker(int id, int numProcs) {
  int matArow;
  MPI_Bcast(&matArow, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int matBcol;
  MPI_Bcast(&matBcol, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int matAcol;
  MPI_Bcast(&matAcol, 1, MPI_INT, 0, MPI_COMM_WORLD);

  float *matA = (float *)malloc(sizeof(float) * matArow * matAcol);
  MPI_Bcast(matA, matArow * matAcol, MPI_FLOAT, 0, MPI_COMM_WORLD);

//  int workSize = (matBcol + numProcs - 1) / numProcs;
  int workSize = (matArow + numProcs - 1) / numProcs;
  int start = workSize * id;
//  int end = start + workSize > matBcol ? matBcol : start + workSize;
  int end = start + workSize > matArow ? matArow : start + workSize;
  printf("Node[%d]: workSize=%d, start=%d, end=%d\n", id, workSize, start, end);
  printf("Node[%d]: matArow=%d, matAcol=%d, matBcol=%d\n", id, matArow, matAcol, matBcol);
  
//  float *matBT = (float *)malloc(sizeof(float) * workSize * matAcol);
//  MPI_Scatter(NULL, workSize*matAcol, MPI_FLOAT, matBT, workSize*matAcol, MPI_FLOAT, 0, MPI_COMM_WORLD);
  float *matBT = (float *)malloc(sizeof(float) * matBcol * matAcol);
  MPI_Bcast(matBT, matBcol*matAcol, MPI_FLOAT, 0, MPI_COMM_WORLD);

  float *matC = (float *)malloc(sizeof(float) * matArow * matBcol);
  memset(matC, 0, sizeof(float) * matArow * matBcol);

  basicSgemm(start, end, 'N', 'T', matArow, matBcol, matAcol, 1.0f, matA, matArow, 
             matBT, matBcol, 0.0f, matC, matArow);

  MPI_Reduce(matC, NULL, matArow*matBcol, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  free(matA);
  free(matBT);
  free(matC);

  MPI_Finalize();
  exit(0);
}

int main (int argc, char *argv[]) {
  struct pb_Parameters *params;
  struct pb_TimerSet timers;
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  MPI_Init(&argc, &argv);
  int numProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  int id;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  if(id) worker(id, numProcs);

  pb_InitializeTimerSet(&timers);
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
  {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
  }
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  readColMajorMatrixFile(params->inpFiles[0], matArow, matAcol, matA);
  readColMajorMatrixFile(params->inpFiles[2], matBcol, matBrow, matBT);
  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  std::vector<float> remote_matC(matArow*matBcol);

  int workSize = (matArow + numProcs - 1) / numProcs;
  int start = workSize * id;
  int end = start + workSize > matArow ? matArow : start + workSize;
  printf("Node[%d]: workSize=%d, start=%d, end=%d\n", id, workSize, start, end);
  printf("Node[%d]: matArow=%d, matAcol=%d, matBcol=%d\n", id, matArow, matAcol, matBcol);
  MPI_Bcast(&matArow, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&matBcol, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&matAcol, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&matA.front(), matArow*matAcol, MPI_FLOAT, 0, MPI_COMM_WORLD);
//  float * remote_B = (float *)malloc(sizeof(float)*matAcol*workSize);
//  MPI_Scatter(&matBT.front(), matAcol*workSize, MPI_FLOAT, remote_B, matAcol*workSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&matBT.front(), matAcol*matBcol, MPI_FLOAT, 0, MPI_COMM_WORLD);

  basicSgemm(start, end, 'N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, 
             &matBT.front(), matBcol, 0.0f, &remote_matC.front(), matArow);

  std::vector<float> matC(matArow*matBcol);
  MPI_Reduce(remote_matC.data(), matC.data(), matArow*matBcol, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (params->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    writeColMajorMatrixFile(params->outFile, matArow, matBcol, matC); 
  }
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  double CPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_COMPUTE]));
  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/CPUtime/1e9 << std::endl;

  MPI_Finalize();

  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  return 0;
}
