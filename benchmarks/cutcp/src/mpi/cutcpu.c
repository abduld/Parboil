//(C) Copyright 2013 University of Illinois, All Rights Reserved

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "atom.h"
#include "cutoff.h"

#undef DEBUG_PASS_RATE
#define CHECK_CYLINDER_CPU

#define CELLEN      4.f
#define INV_CELLEN  (1.f/CELLEN)

extern int mpi_compute_cutoff_potential_lattice(
    int begin,		// begin of the grid index
    int end,		// end of the grid index
    LatticeDim dim,	// size of the lattice
    float *lattice,	// the lattice array
    float cutoff,	// cutoff distance
    Atom *atom,		// array of atoms
    int natoms,		// number of atoms in the array
    int nxcell,		// number of cells in x dimention
    int nycell,		// number of cells in y dimention
    int nzcell		// number of cells in z dimention
    )
{

  int nx = dim.nx;
  int ny = dim.ny;
  int nz = dim.nz;
  float xlo = dim.lo.x;
  float ylo = dim.lo.y;
  float zlo = dim.lo.z;
  float gridspacing = dim.h;


  const float a2 = cutoff * cutoff;
  const float inv_a2 = 1.f / a2;
  float s;
  const float inv_gridspacing = 1.f / gridspacing;
  const int radius = (int) ceilf(cutoff * inv_gridspacing) - 1;

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
  int ncell;
  Vec3 minext, maxext;
  int *first, *next;
  float inv_cellen = INV_CELLEN;

#if DEBUG_PASS_RATE
  unsigned long long pass_count = 0;
  unsigned long long fail_count = 0;
#endif

  Atoms* atoms = (Atoms *)malloc(sizeof(Atoms));
  atoms->atoms = atom;
  atoms->size = natoms;
  get_atom_extent(&minext, &maxext, atoms);
  ncell = nxcell * nycell * nzcell;
  first = (int *) malloc(ncell * sizeof(int));
  for (gindex = 0;  gindex < ncell;  gindex++) {
    first[gindex] = -1;
  }
  next = (int *) malloc(natoms * sizeof(int));
  for (n = 0;  n < natoms;  n++) {
    next[n] = -1;
  }

  for (n = 0;  n < natoms;  n++) {
    if (0==atom[n].q) continue;
    i = (int) floorf((atom[n].x - minext.x) * inv_cellen);
    j = (int) floorf((atom[n].y - minext.y) * inv_cellen);
    k = (int) floorf((atom[n].z - minext.z) * inv_cellen);
    gindex = (k*nycell + j)*nxcell + i;
    next[n] = first[gindex];
    first[gindex] = n;
  }

  for(gindex=begin; gindex<end; gindex++) {
    for (n = first[gindex];  n != -1;  n = next[n]) {
      x = atom[n].x - xlo;
      y = atom[n].y - ylo;
      z = atom[n].z - zlo;
      q = atom[n].q;

//find closest grid point with position less than or equal to atom
      ic = (int) (x * inv_gridspacing);
      jc = (int) (y * inv_gridspacing);
      kc = (int) (z * inv_gridspacing);

//find extent of surrounding box of grid points
      ia = ic - radius;
      ib = ic + radius + 1;
      ja = jc - radius;
      jb = jc + radius + 1;
      ka = kc - radius;
      kb = kc + radius + 1;

//trim box edges so that they are within grid point lattice
      if (ia < 0)   ia = 0;
      if (ib >= nx) ib = nx-1;
      if (ja < 0)   ja = 0;
      if (jb >= ny) jb = ny-1;
      if (ka < 0)   ka = 0;
      if (kb >= nz) kb = nz-1;

//loop over surrounding grid points
      xstart = ia*gridspacing - x;
      ystart = ja*gridspacing - y;
      dz = ka*gridspacing - z;
      for (k = ka;  k <= kb;  k++, dz += gridspacing) {
        koff = k*ny;
        dz2 = dz*dz;
        dy = ystart;
        for (j = ja;  j <= jb;  j++, dy += gridspacing) {
          jkoff = (koff + j)*nx;
          dydz2 = dy*dy + dz2;
#ifdef CHECK_CYLINDER_CPU
          if (dydz2 >= a2) continue;
#endif

          dx = xstart;
          index = jkoff + ia;
          pg = lattice + index;

          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
            r2 = dx*dx + dydz2;
            if (r2 >= a2) {
		  continue;
            }
            s = (1.f - r2 * inv_a2);
            e = q * (1/sqrtf(r2)) * s * s;
            *pg += e;
          }// end for i
        }// end for j
      } // end for k: loop over surrounding grid points
    } // end for n: loop over atoms in a gridcell
  } // end for gindex: loop over gridcells


  free(next);
  free(first);
  free(atoms);
  return 0;
}
