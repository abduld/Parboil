/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2012 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "atom.h"
#include "cutoff.h"

#define CELLEN      4.f
#define INV_CELLEN  (1.f/CELLEN)

/* This is a naive, algorithmically inefficient algorithm that
 * visits every grid point for every atom..
 * Because it is so simple, it is useful as a reference.
 */
extern int cpu_compute_cutoff_potential_lattice(
    Lattice *lattice,                  /* the lattice */
    float cutoff,                      /* cutoff distance */
    Atoms *atoms                       /* array of atoms */
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

  int n;
  int i, j, k;

  float x, y, z, q;
  float dx, dy, dz;
  float r2;
  float e;

  float *pg;

  /* For each atom */
  for (n = 0; n < natoms; n++) {
    q = atom[n].q;
    if (q == 0) continue;  /* skip any non-contributing atoms */
    x = atom[n].x - xlo;
    y = atom[n].y - ylo;
    z = atom[n].z - zlo;

    /* For each grid point */
    pg = lattice->lattice;
    dz = -z;
    for (k = 0; k < nz; k++, dz += gridspacing) {
      dy = -y;
      for (j = 0; j < ny; j++, dy += gridspacing) {
        dx = -x;
        for (i = 0; i < nx; i++, pg++, dx += gridspacing) {
          /* Add electrostatic potential at this grid point */
          r2 = dz*dz + dy*dy + dx*dx;
          if (r2 >= a2) continue; /* Skip if not within cutoff radius */
          s = (1.f - r2 * inv_a2);
          e = q * (1/sqrtf(r2)) * s * s;
          *pg += e;
        }
      }
    }
  }

  return 0;
}
