// (C) Copyright 2013, University of Illinois, All Rights Reserved

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include "arg.h"
#include "model.h"
///*
static void worker(int id, int numProcs) {
//  printf("Node[%d]: ready to work\n", id);
  int i;
  int rand_num;
  MPI_Bcast(&rand_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int workSize = (rand_num + numProcs - 1) / numProcs;
  int start = workSize * id;
  int end = start + workSize > rand_num ? rand_num : start + workSize;

  int num_points;
  MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int nbins;
  MPI_Bcast(&nbins, 1, MPI_INT, 0, MPI_COMM_WORLD);
  float * binb = (float *) malloc(sizeof(float)*(nbins+1));
  MPI_Bcast(binb, nbins+1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int npd;
  MPI_Bcast(&npd, 1, MPI_INT, 0, MPI_COMM_WORLD);
  struct cartesian * data = (struct cartesian*)malloc(sizeof(struct cartesian)*num_points);
  MPI_Bcast(data, num_points * sizeof(struct cartesian) / sizeof(float), MPI_FLOAT, 0, MPI_COMM_WORLD);

  int *remote_npr = (int *) malloc(sizeof(int)*rand_num);
  MPI_Scatter(NULL, workSize, MPI_INT, remote_npr, workSize, MPI_INT, 0, MPI_COMM_WORLD);
  size_t rand_size = sizeof(struct cartesian)*num_points*workSize;
  int count = rand_size/sizeof(float);
  struct cartesian * remote_random = (struct cartesian*)malloc(rand_size);
  MPI_Scatter(NULL, count, MPI_FLOAT, remote_random, count, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int memsize = (nbins+2)*sizeof(unsigned long long);
  unsigned long long* RRS = (unsigned long long*)malloc(memsize);
  bzero(RRS, memsize);
  unsigned long long* DRS = (unsigned long long*)malloc(memsize);
  bzero(DRS, memsize);

  for (i = 0; i < end-start; i++) {
    struct cartesian * index = remote_random+i*num_points;
    doCompute(index, remote_npr[i], NULL, 0, 1, RRS, nbins, binb);
    doCompute(data, npd, index, remote_npr[i], 0, DRS, nbins, binb);
  }

  int reduce_count = memsize/sizeof(unsigned long long);
  MPI_Reduce(DRS, NULL, reduce_count, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(RRS, NULL, reduce_count, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  
  free(data);
  free(remote_random);
  free(remote_npr);
  free(binb);
  free(DRS);
  free(RRS);
  MPI_Finalize();
  exit(0);
}

int main( int argc, char **argv ) {
  struct pb_TimerSet timers;
  struct pb_Parameters *params;
  int rf, k, nbins, npd;
  int *npr;
  float *binb;
  unsigned long long *DD, *remote_RRS, *remote_DRS;
  size_t memsize;
  struct cartesian *data;
  struct cartesian *random;
  FILE *outfile;
  int rand_num;
  int num_points;

  MPI_Init(&argc, &argv);
  int numProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  int id;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  if(id) worker(id, numProcs);


  pb_InitializeTimerSet( &timers );
  params = pb_ReadParameters( &argc, argv );
  options args;
  parse_args( argc, argv, &args );
  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  nbins = (int)floor(bins_per_dec * (log10(max_arcmin) - log10(min_arcmin)));
  memsize = (nbins+2)*sizeof(unsigned long long);
    
  // memory for bin boundaries
  binb = (float *)malloc((nbins+1)*sizeof(float));
  if (binb == NULL) {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
  }
  for (k = 0; k < nbins+1; k++) {
      binb[k] = cos(pow(10, log10(min_arcmin) + k*1.0/bins_per_dec) / 60.0*D2R);
  }

  // memory for DD
  DD = (unsigned long long*)malloc(memsize);
  if (DD == NULL) {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
  }
  bzero(DD, memsize);
    
  // memory for RR
  remote_RRS = (unsigned long long*)malloc(memsize);
  if (remote_RRS == NULL) {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
  }
  bzero(remote_RRS, memsize);
    
  // memory for DR
  remote_DRS = (unsigned long long*)malloc(memsize);
  if (remote_DRS == NULL) {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
  }
  bzero(remote_DRS, memsize);

  // memory for input data
  num_points = args.npoints;
  data = (struct cartesian*)malloc(num_points * sizeof(struct cartesian));
  if (data == NULL) {
      fprintf(stderr, "Unable to allocate memory for % data points (#1)\n", num_points);
      return(0);
  }

  rand_num = args.random_count;

  // read data file
  pb_SwitchToTimer( &timers, pb_TimerID_IO );
  npd = readdatafile(params->inpFiles[0], data, num_points);
  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  if (npd != num_points) {
      fprintf(stderr, "Error: read %i data points out of %i\n", npd, num_points);
      return(0);
  }

  doCompute(data, npd, NULL, 0, 1, DD, nbins, binb);

  npr = (int *) malloc(rand_num * sizeof(int));
  random = (struct cartesian*)malloc(rand_num * num_points * sizeof(struct cartesian));
  if (random == NULL) {
    fprintf(stderr, "Unable to allocate memory for % data points (#2)\n", num_points);
    return(0);
  }

  for (rf = 0; rf < rand_num; rf++) {
    pb_SwitchToTimer( &timers, pb_TimerID_IO );
    npr[rf] = readdatafile(params->inpFiles[rf+1], random+rf*num_points, num_points);

    pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
    if (npr[rf] != num_points) {
      fprintf(stderr, "Error: read %i random points out of %i in file %s\n", npr[rf], num_points, params->inpFiles[rf+1]);
      return(0);
    }
  }

  int workSize = (rand_num + numProcs - 1) / numProcs;
  int start = workSize * id;
  int end = start + workSize > rand_num ? rand_num : start + workSize;

  MPI_Bcast(&rand_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nbins, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(binb, nbins+1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&npd, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(data, num_points * sizeof(struct cartesian) / sizeof(float), MPI_FLOAT, 0, MPI_COMM_WORLD);

  int *remote_npr = (int *) malloc(sizeof(int)*rand_num);
  MPI_Scatter(npr, workSize, MPI_INT, remote_npr, workSize, MPI_INT, 0, MPI_COMM_WORLD);
  size_t rand_size = sizeof(struct cartesian)*num_points*workSize;
  int count = rand_size/sizeof(float);
  struct cartesian * remote_random = (struct cartesian*)malloc(rand_size);
  MPI_Scatter(random, count, MPI_FLOAT, remote_random, count, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // loop through random data files
  for (rf = start; rf < end; rf++) {
    struct cartesian * index = remote_random+rf*num_points;
    doCompute(index, remote_npr[rf], NULL, 0, 1, remote_RRS, nbins, binb);
    doCompute(data, npd, index, remote_npr[rf], 0, remote_DRS, nbins, binb);
  }

  unsigned long long *DRS = (unsigned long long*)malloc(memsize);
  if (DRS == NULL) {
      printf("Unable to allocate memory\n");
      exit(-1);
  }
  bzero(DRS, memsize);
  unsigned long long *RRS = (unsigned long long*)malloc(memsize);
  if (RRS == NULL) {
      printf("Unable to allocate memory\n");
      exit(-1);
  }
  bzero(RRS, memsize);

  int reduce_count = memsize/sizeof(unsigned long long);
  MPI_Reduce(remote_DRS, DRS, reduce_count, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(remote_RRS, RRS, reduce_count, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if ((outfile = fopen(params->outFile, "w")) == NULL) {
      fprintf(stderr, "Unable to open output file %s for writing, assuming stdout\n", params->outFile);
      outfile = stdout;
  }
  pb_SwitchToTimer( &timers, pb_TimerID_IO );
  for (k = 1; k < nbins+1; k++) {
      fprintf(outfile, "%llu\n%llu\n%llu\n", DD[k], DRS[k], RRS[k]);      
  }
  if(outfile != stdout)
    fclose(outfile);

  // free memory
  free(data);
  free(random);
  free(binb);
  free(DD);
  free(RRS);
  free(DRS);
  free(remote_RRS);
  free(remote_DRS);
  free(npr);
  free(remote_npr);
  free(remote_random);
  MPI_Finalize();

  pb_SwitchToTimer( &timers, pb_TimerID_NONE );
  pb_PrintTimerSet( &timers );
  pb_FreeParameters( params );
  return 0;
}

