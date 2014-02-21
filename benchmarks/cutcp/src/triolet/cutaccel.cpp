/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <TrioletData.h>
#include <sys/time.h>
#include "atom.h"
#include "cutoff.h"
#include "kernel_cxx.h"

#define CHECK_CYLINDER_CPU

#define CELLEN      4.f
#define INV_CELLEN  (1.f/CELLEN)

using namespace Triolet;

// A naive, sequential, loop-based implementation of the cutoff loop
static void
naive_cutoff_loop(float *lattice, int nz, int ny, int nx,
                  float zlo, float ylo, float xlo,
                  Atom *atom,
                  int natoms,
                  float gridspacing,
                  float cutoff)
{
  float a2 = cutoff * cutoff;
  float inv_a2 = 1 / a2;
  float *pg = lattice;
  int k;
  for (k = 0; k < nz; k++) {
    float z = zlo + gridspacing * k;
    int j;
    for (j = 0; j < ny; j++) {
      float y = ylo + gridspacing * j;
      int i;
      for (i = 0; i < nx; i++) {
        float x = xlo + gridspacing * i;
        float e = 0;
        int index;
        for (index = 0; index < natoms; index++) {
          float dx = atom[index].x - x;
          float dy = atom[index].y - y;
          float dz = atom[index].z - z;
          float q = atom[index].q;
          float r2 = dx*dx + dy*dy + dz*dz;
          if (r2 < a2) {
            float s = (1.f - r2 * inv_a2) * (1.f - r2 * inv_a2);
            e += q * (1/sqrtf(r2)) * s;
          }
        }
        *pg++ = e;
      }
    }
  }
}

extern int accel_compute_cutoff_potential_lattice(
    Lattice *lattice,                  /* the lattice */
    float cutoff,                      /* cutoff distance */
    Atoms *atoms,                      /* array of atoms */
    struct pb_TimerSet *timers
    )
{
  int nx = lattice->dim.nx;
  int ny = lattice->dim.ny;
  int nz = lattice->dim.nz;
  float xlo = lattice->dim.lo.x;
  float ylo = lattice->dim.lo.y;
  float zlo = lattice->dim.lo.z;
  float gridspacing = lattice->dim.h;
  int natoms = atoms->size;
  Atom *atom = atoms->atoms;

  const float a2 = cutoff * cutoff;
  const float inv_a2 = 1.f / a2;
  float s;
  const float inv_gridspacing = 1.f / gridspacing;
  const int radius = (int) ceilf(cutoff * inv_gridspacing) - 1;
    /* lattice point radius about each atom */

  int n;
  int i, j, k;
  int ia, ib, ic;
  int ja, jb, jc;
  int ka, kb, kc;
  int index;
  int koff, jkoff;

  float x, y, z, q;
  float dx, dy, dz;
  float dz2, dydz2, r2;
  float e;
  float xstart, ystart;

  float *pg;

  int gindex;
  int ncell, nxcell, nycell, nzcell;
  int *first, *next;
  float inv_cellen = INV_CELLEN;
  Vec3 minext, maxext;		/* Extent of atom bounding box */
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;

  /* find min and max extent */
  get_atom_extent(&minext, &maxext, atoms);


#if 0
  naive_cutoff_loop(lattice->lattice, nz, ny, nx, zlo, ylo, xlo,
                    atom, natoms, gridspacing, cutoff);
#else

  pb_SwitchToTimer(timers, pb_TimerID_COPY);

  /* Marshal data */
  Incomplete<List<Tuple<Float, Float, Float, Float> > > mk_atoms;
  mk_atoms.create(natoms);
  {
    int i;
    for (i = 0; i < natoms; i++) {
      mk_atoms.at(i).get<0>() = (Float)atom[i].x;
      mk_atoms.at(i).get<1>() = (Float)atom[i].y;
      mk_atoms.at(i).get<2>() = (Float)atom[i].z;
      mk_atoms.at(i).get<3>() = (Float)atom[i].q;
    }
  }
  List<Tuple<Float, Float, Float, Float> > accel_atoms = mk_atoms.freeze();

  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);

  /* Run kernel */
  Array3<Float> accel_potential =
    cutoff_kernel(nz, ny, nx, zlo, ylo, xlo,
                        gridspacing, cutoff, accel_atoms);

  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  /* Marshal data */
  {
    float *pg = lattice->lattice;
    int x, y, z;
    for (z = 0; z < nz; z++)
      for (y = 0; y < ny; y++)
        for (x = 0; x < nx; x++)
          *pg++ = (float)(Float)accel_potential.at(z, y, x);
  }

  pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
#endif

  return 0;
}
