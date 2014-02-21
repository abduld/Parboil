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
#include "compute_cxx.h"


// Convert the cartesian data to three Triolet lists.
tri_cartesian cartesian_to_arrays(struct cartesian *data, int size)
{
  Incomplete<tri_cartesian> output;
  output.create(size);

  int i;
  for (i = 0; i < size; i++) {
    output.at(i).get<0>() = (Float)data[i].x;
    output.at(i).get<1>() = (Float)data[i].y;
    output.at(i).get<2>() = (Float)data[i].z;
  }
  return output.freeze();
}

tri_cartesian_dataset cartesian_to_arrays(struct cartesian *data, int size, int nsets) {
  Incomplete<tri_cartesian_dataset> output;
  output.create(nsets);
  for (int ii = 0; ii < nsets; ii++) {
      struct cartesian * d = &data[ii * size];
      Incomplete<tri_cartesian> a = output.at(ii);
      a.initialize(size);
      for (int jj = 0; jj < size; jj++) {
        a.at(jj).get<0>() = (Float)d[jj].x;
        a.at(jj).get<1>() = (Float)d[jj].y;
        a.at(jj).get<2>() = (Float)d[jj].z;
      }
      output.at(ii) = a;
  }
  return output.freeze();
}

int doComputeCross(tri_cartesian_dataset x, tri_cartesian_dataset y,
                   long long *data_bins, int nbins, int nsets, float *binb,
                   struct pb_TimerSet *timers)
{
  int ii;

  // Setup
  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  List<Float> binb_list = CreateFloatList(1 + nbins, binb);

  // Compute
  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  List<Int64> histo = compute_cross(x.at(0), y, binb_list, nbins, nsets);

  // Copy output data into main histogram
  pb_SwitchToTimer(timers, pb_TimerID_COPY);

  for (ii = 0; ii < nbins+1; ii++) {
    data_bins[ii] = histo.at(ii);
  }

  pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
  return 0;
}

int doComputeSelf(tri_cartesian_dataset x,
                  long long *data_bins, int nbins, int nsets, float *binb,
                  struct pb_TimerSet *timers)
{

  int ii;

  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  List<Float> binb_list = CreateFloatList(1 + nbins, binb);

  // Compute
  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  List<Int64> histo = compute_self(x, binb_list, nbins, nsets);

  // Copy output data into main histogram
  pb_SwitchToTimer(timers, pb_TimerID_COPY);

  for (ii = 0; ii < nbins+1; ii++) {
    data_bins[ii] = histo.at(ii);
  }

  pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
  return 0;
}


