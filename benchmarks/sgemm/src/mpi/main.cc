// (C) Copyright 2013, University of Illinois. All Rights Reserved.
// Xuhao Chen <cxh.nudt@gmail.com>
// MPI implementation

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
#define WORK_TAG 1
#define DIE_TAG 2
#define MASTER 0

extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

static void worker(int id, int numProcs) {
  int matArow;
  MPI_Bcast(&matArow, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int matBcol;
  MPI_Bcast(&matBcol, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int matAcol;
  MPI_Bcast(&matAcol, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int workSize = (matArow + numProcs - 1) / numProcs;
  float *matA = (float *)malloc(sizeof(float) * workSize * matAcol);
  MPI_Recv(matA, workSize*matAcol, MPI_FLOAT, MASTER, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  float *matBT = (float *)malloc(sizeof(float) * matBcol * matAcol);
  MPI_Bcast(matBT, matBcol*matAcol, MPI_FLOAT, 0, MPI_COMM_WORLD);
  float *matC = (float *)malloc(sizeof(float) * workSize * matBcol);
  memset(matC, 0, sizeof(float) * workSize * matBcol);
  basicSgemm('N', 'T', workSize, matBcol, matAcol, 1.0f, matA, workSize, matBT, matBcol, 0.0f, matC, workSize);
  MPI_Send(matC, workSize*matBcol, MPI_FLOAT, MASTER, WORK_TAG, MPI_COMM_WORLD);
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
      || (params->inpFiles[3] != NULL)) {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
  }
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  readColMajorMatrixFile(params->inpFiles[0], matArow, matAcol, matA);
  readColMajorMatrixFile(params->inpFiles[2], matBcol, matBrow, matBT);
  double startTime, endTime;
  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  startTime = MPI_Wtime();
  std::vector<float> matC(matArow*matBcol);

  float *A = matA.data();
  float *B = matBT.data();
  float *A2 = (float *)malloc(matArow * matAcol * sizeof(float));
  float *B2 = (float *)malloc(matBrow * matBcol * sizeof(float));
  {
    for (int i = 0; i < matAcol; i++) {
      for (int j = 0; j < matArow; j++) {
        A2[j*matAcol+i] = A[j+i*matArow];
      }
    }
    for (int i = 0; i < matBrow; i++) {
      for (int j = 0; j < matBcol; j++) {
        B2[j*matBrow+i] = B[j+i*matBcol];
      }
    }
  }
  int workSize = (matArow + numProcs - 1) / numProcs;
  MPI_Bcast(&matArow, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&matBcol, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&matAcol, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int i;
  int numElements = workSize*matAcol;
  int offset = numElements;
  for(i=1;i<numProcs;i++) {
    MPI_Send(A2+offset, numElements, MPI_FLOAT, i, WORK_TAG, MPI_COMM_WORLD);
    offset += numElements;
  }
  MPI_Bcast(B2, matBrow*matBcol, MPI_FLOAT, 0, MPI_COMM_WORLD);
  float * C = matC.data();
  float * C2 = (float *)malloc(matArow * matBcol * sizeof(float));

  basicSgemm('N', 'T', workSize, matBcol, matAcol, 1.0f, A2, workSize, B2, matBcol, 0.0f, C2, workSize);

  numElements = workSize*matBcol;
  offset = numElements;
  for(i=1;i<numProcs;i++) {
    MPI_Recv(C2+offset, numElements, MPI_FLOAT, i, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    offset += numElements;
  }
  for (int i = 0; i < matArow; i++) {
    for (int j = 0; j < matBcol; j++) {
      C[j*matArow+i] = C2[j+i*matBcol];
    }
  }
  endTime = MPI_Wtime();
  printf("MPI Timing is %f\n", endTime-startTime);
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
