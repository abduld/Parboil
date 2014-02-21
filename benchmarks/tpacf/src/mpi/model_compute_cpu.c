// (C) Copyright 2013, University of Illinois, All Rights Reserved
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <stdio.h> 
#include "model.h"

int doCompute(struct cartesian *data1, int n1, struct cartesian *data2, 
	      int n2, int doSelf, unsigned long long *data_bins, int nbins, float *binb) {
  int i, j;
  if (doSelf) {
      n2 = n1;
      data2 = data1;
  }
//  printf("Loop over data 1, n1=%d, n2=%d, nbins=%d\n", n1, n2, nbins);
  for (i = 0; i < ((doSelf) ? n1-1 : n1); i++) {
      const register float xi = data1[i].x;
      const register float yi = data1[i].y;
      const register float zi = data1[i].z;
      
//      printf("Loop over data 2\n");
      for (j = ((doSelf) ? i+1 : 0); j < n2; j++) {
	  register float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;
	  register int min = 0;
	  register int max = nbins;
	  register int k;
	  
	  while (max > min+1) {
	      k = (min + max) / 2;
	      if (dot >= binb[k]) 
		max = k;
	      else 
		min = k;
          };
	  
	  if (dot >= binb[min]) {
	      data_bins[min] += 1; 
          }
	  else if (dot < binb[max]) { 
	      data_bins[max+1] += 1; 
          }
	  else { 
	      data_bins[max] += 1;
          }
      } // end for j
  } // end for i
  return 0;
}

