/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifndef __MODEL_H__
#define __MODEL_H__

#include <parboil.h>
#include "TrioletData.h"

using namespace Triolet;
typedef List<Tuple<Float, Float, Float> > tri_cartesian;
typedef List<tri_cartesian> tri_cartesian_dataset;

#define D2R M_PI/180.0
#define R2D 180.0/M_PI
#define R2AM 60.0*180.0/M_PI

#define bins_per_dec 5
#define min_arcmin 1.0
#define max_arcmin 10000.0

#define NUM_BINS 20

typedef unsigned long hist_t;

struct spherical
{
  float ra, dec;  // latitude, longitude pair
};

struct cartesian
{
  float x, y, z;  // cartesian coodrinates
};

int readdatafile(char *fname, struct cartesian *data, int npoints);

tri_cartesian cartesian_to_arrays(struct cartesian *data, int size);
tri_cartesian_dataset cartesian_to_arrays(struct cartesian *data, int size, int nsets);

int doComputeSelf(tri_cartesian_dataset x,
                   long long *data_bins,
                  int nbins, int nsets, float *binb, struct pb_TimerSet *timers);

int doComputeCross(tri_cartesian_dataset x, tri_cartesian_dataset y,
	      long long *data_bins,
                   int nbins, int nsets, float *binb, struct pb_TimerSet *timers);


void initBinB(struct pb_TimerSet *timers);

#endif
