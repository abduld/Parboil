/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

#include "args.h"
#include "model.h"


int main( int argc, char **argv )
{
  struct pb_TimerSet timers;
  struct pb_Parameters *params;
  int rf, k, nbins, npd, * npr;
  float *binb, w;
  long long *DD, *RRS, *DRS;
  size_t memsize;
  struct cartesian *data, *random;
  FILE *outfile;
  int offset = 0;

  Triolet_init(&argc, &argv);

  pb_InitializeTimerSet( &timers );
  params = pb_ReadParameters( &argc, argv );
  options args;
  parse_args( argc, argv, &args );

  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  nbins = (int)floor(bins_per_dec * (log10(max_arcmin) -
					 log10(min_arcmin)));
  memsize = (nbins+2)*sizeof(long long);

  // memory for bin boundaries
  binb = (float *)malloc((nbins+1)*sizeof(float));
  if (binb == NULL)
    {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
    }
  for (k = 0; k < nbins+1; k++)
    {
      binb[k] = cos(pow(10, log10(min_arcmin) +
			k*1.0/bins_per_dec) / 60.0*D2R);
      printf("%.10f\n", binb[k]);
    }

  // memory for DD
  DD = (long long*)malloc(memsize);
  if (DD == NULL)
    {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
    }
  bzero(DD, memsize);

  // memory for RR
  RRS = (long long*)malloc(memsize);
  if (RRS == NULL)
    {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
    }
  bzero(RRS, memsize);

  // memory for DR
  DRS = (long long*)malloc(memsize);
  if (DRS == NULL)
    {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
    }
  bzero(DRS, memsize);

  // memory for input data
  data = (struct cartesian*)malloc
    (args.npoints* sizeof(struct cartesian));
  if (data == NULL)
    {
      fprintf(stderr,
	      "Unable to allocate memory for % data points (#1)\n",
	      args.npoints);
      return(0);
    }

  random = (struct cartesian*)malloc
    (args.npoints*sizeof(struct cartesian));
  if (random == NULL)
    {
      fprintf(stderr,
	      "Unable to allocate memory for % data points (#2)\n",
	      args.npoints);
      return(0);
    }

  printf("Min distance: %f arcmin\n", min_arcmin);
  printf("Max distance: %f arcmin\n", max_arcmin);
  printf("Bins per dec: %i\n", bins_per_dec);
  printf("Total bins  : %i\n", nbins);

  // read data file
  pb_SwitchToTimer( &timers, pb_TimerID_IO );
  npd = readdatafile(params->inpFiles[0], data, args.npoints);
  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  if (npd != args.npoints)
    {
      fprintf(stderr,
	      "Error: read %i data points out of %i\n",
	      npd, args.npoints);
      return(0);
    }

  // Marshal to Pyon
  tri_cartesian_dataset pyon_data = cartesian_to_arrays(data, npd, 1);

  // compute DD
  doComputeSelf(pyon_data, DD, nbins, 1, binb, &timers);

  npr = (int *) malloc(sizeof(int) * args.npoints);
  assert(npr != NULL);

  free(random);
  random = (struct cartesian *) malloc(sizeof(struct cartesian) * args.npoints * args.random_count);

  // loop through random data files
  for (rf = 0; rf < args.random_count; rf++)
    {
      // read random file
      pb_SwitchToTimer( &timers, pb_TimerID_IO );
      npr[rf] = readdatafile(params->inpFiles[rf+1], &random[offset], args.npoints);
      pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
      offset += npr[rf];
      if (npr[rf] != args.npoints)
        {
	  fprintf(stderr,
		  "Error: read %i random points out of %i in file %s\n",
		  npr[rf], args.npoints, params->inpFiles[rf+1]);
	  return(0);
        }

    }
  // compute RR
  tri_cartesian_dataset pyon_random = cartesian_to_arrays(random, npr[0], args.random_count);
  doComputeSelf(pyon_random, RRS, nbins, args.random_count, binb, &timers);

  // compute DR
  doComputeCross(pyon_data, pyon_random, DRS, nbins, args.random_count, binb, &timers);

  // compute and output results
  if ((outfile = fopen(params->outFile, "w")) == NULL)
    {
      fprintf(stderr,
	      "Unable to open output file %s for writing, assuming stdout\n",
	      params->outFile);
      outfile = stdout;
    }

  pb_SwitchToTimer( &timers, pb_TimerID_IO );
  for (k = 1; k < nbins+1; k++)
    {
      fprintf(outfile, "%lld\n%lld\n%lld\n", DD[k], DRS[k], RRS[k]);
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
  free(npr);

  pb_SwitchToTimer( &timers, pb_TimerID_NONE );
  pb_PrintTimerSet( &timers );
  pb_FreeParameters( params );
}

