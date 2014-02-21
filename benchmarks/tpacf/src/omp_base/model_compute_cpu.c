/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <stdio.h> 
#include <stdlib.h>
#include "model.h"

int doCompute(struct cartesian *data1, int n1, struct cartesian *data2, 
	      int n2, int doSelf, long long *data_bins, 
	      int nbins, float *binb)
{
    int i, j;
    if (doSelf) {
      n2 = n1;
      data2 = data1;
    }

    int num_threads = omp_get_max_threads();
    long long *private_data_bins = (long long *) malloc(sizeof(long long)*(nbins+2)*num_threads);
    memset(private_data_bins,0,(nbins+2)*num_threads*sizeof(long long));
	
#pragma omp parallel for schedule(static, 4)
    for (i = 0; i < ((doSelf) ? n1-1 : n1); i++) {
      const register float xi = data1[i].x;
      const register float yi = data1[i].y;
      const register float zi = data1[i].z;

      for (j = ((doSelf) ? i+1 : 0); j < n2; j++) {
	  register float dot = xi * data2[j].x + yi * data2[j].y + 
	  zi * data2[j].z;
	  
	  // run binary search
	  register int min = 0;
	  register int max = nbins;
	  register int k, indx;
	  
	  while (max > min+1) {
	      k = (min + max) / 2;
	      if (dot >= binb[k]) 
		max = k;
	      else 
		min = k;
          };
	  
	  int tid = omp_get_thread_num();
	  if (dot >= binb[min]) {
	      private_data_bins[tid*(nbins+2)+min]+=1;
	  }
	  else if (dot < binb[max]) {
	      private_data_bins[tid*(nbins+2)+max+1]+=1;
	  }
	  else {
	      private_data_bins[tid*(nbins+2)+max]+=1;
	  }
       }//end for j
    }//end for i

	int m,n;
//	#pragma omp parallel for
    for(m=0; m<num_threads; m++) {
      for(n=0; n<(nbins+2); n++) {
        data_bins[n] += private_data_bins[m*(nbins+2)+n];
      }
    }
    free(private_data_bins);
    return 0;
}

