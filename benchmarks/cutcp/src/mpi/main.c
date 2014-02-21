//(C) Copyright 2013 University of Illinois, All Rights Reserved

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "parboil.h"
#include "atom.h"
#include "cutoff.h"
#include "output.h"

#define ERRTOL 1e-4f
#define NOKERNELS             0
#define CUTOFF1               1
#define CUTOFF6              32
#define CUTOFF6OVERLAP       64
#define CUTOFFCPU         16384

int appenddata(const char *filename, int size, double time) {
  FILE *fp;
  fp=fopen(filename, "a");
  if (fp == NULL) {
    printf("error appending to file %s..\n", filename);
    return -1;
  }
  fprintf(fp, "%d  %.3f\n", size, time);
  fclose(fp);
  return 0;
}

LatticeDim lattice_from_bounding_box(Vec3 lo, Vec3 hi, float h) {
  LatticeDim ret;
  ret.nx = (int) floorf((hi.x-lo.x)/h) + 1;
  ret.ny = (int) floorf((hi.y-lo.y)/h) + 1;
  ret.nz = (int) floorf((hi.z-lo.z)/h) + 1;
  ret.lo = lo;
  ret.h = h;
  return ret;
}

Lattice * create_lattice(LatticeDim dim) {
  int size;
  Lattice *lat = (Lattice *)malloc(sizeof(Lattice));

  if (lat == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  lat->dim = dim;

  /* Round up the allocated size to a multiple of 8 */
  size = ((dim.nx * dim.ny * dim.nz) + 7) & ~7;
  lat->lattice = (float *)calloc(size, sizeof(float));

  if (lat->lattice == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  return lat;
}


void destroy_lattice(Lattice *lat) {
  if (lat) {
    free(lat->lattice);
    free(lat);
  }
}

static void worker(int id, int numProcs) {
  float cutoff;
  int numAtoms;
  MPI_Bcast(&numAtoms, 1, MPI_INT, 0, MPI_COMM_WORLD);
  Atom* atom_array;
  atom_array = (Atom*)malloc(numAtoms*sizeof(Atom));
  MPI_Bcast(atom_array, numAtoms*sizeof(Atom)/sizeof(float), MPI_FLOAT, 0, MPI_COMM_WORLD);

  LatticeDim lattice_dim;
  MPI_Bcast(&lattice_dim, sizeof(LatticeDim)/sizeof(float), MPI_FLOAT, 0, MPI_COMM_WORLD);
  int lattice_size = ((lattice_dim.nx * lattice_dim.ny * lattice_dim.nz) + 7) & ~7;
  float *remote_lattice = (float*) calloc(lattice_size, sizeof(float));

  MPI_Bcast(&cutoff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);


  int nxcell;
  MPI_Bcast(&nxcell, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int nycell;
  MPI_Bcast(&nycell, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int nzcell;
  MPI_Bcast(&nzcell, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int ncell = nxcell*nycell*nzcell;

  int workSize = (ncell + numProcs - 1) / numProcs;
  int start = workSize * id;
  int end = start + workSize > ncell ? ncell : start + workSize;

  int i;
  if (mpi_compute_cutoff_potential_lattice(start, end, lattice_dim, remote_lattice, cutoff, atom_array, numAtoms, nxcell, nycell, nzcell)) {
    fprintf(stderr, "Computation failed\n");
    exit(1);
  }

  MPI_Reduce(remote_lattice, NULL, lattice_size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  free(atom_array);
  free(remote_lattice);

  MPI_Finalize();
  exit(0);
}

int main(int argc, char *argv[]) {
  Atoms *atom;
  LatticeDim lattice_dim;
  Lattice * lattice;
  Vec3 min_ext, max_ext;	/* Bounding box of atoms */
  Vec3 lo, hi;			/* Bounding box with padding  */
  float h = 0.5f;		/* Lattice spacing */
  float cutoff = 12.f;		/* Cutoff radius */
  float exclcutoff = 1.f;	/* Radius for exclusion */
  float padding = 0.5f;		/* Bounding box padding distance */
  int n;

  struct pb_Parameters *parameters;
  struct pb_TimerSet timers;

  MPI_Init(&argc, &argv);
  int numProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  int id;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  if(id) worker(id, numProcs);


  parameters = pb_ReadParameters(&argc, argv);
  if (parameters == NULL) {
    exit(1);
  }
  if (pb_Parameters_CountInputs(parameters) != 1) {
    fprintf(stderr, "Expecting one input file\n");
    exit(1);
  }
  pb_InitializeTimerSet(&timers);
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  {
    const char *pqrfilename = parameters->inpFiles[0];
    if (!(atom = read_atom_file(pqrfilename))) {
      fprintf(stderr, "read_atom_file() failed\n");
      exit(1);
    }
    printf("read %d atoms from file '%s'\n", atom->size, pqrfilename);
  }
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  get_atom_extent(&min_ext, &max_ext, atom);
  printf("extent of domain is:\n");
  printf("  minimum %g %g %g\n", min_ext.x, min_ext.y, min_ext.z);
  printf("  maximum %g %g %g\n", max_ext.x, max_ext.y, max_ext.z);

  printf("padding domain by %g Angstroms\n", padding);
  lo = (Vec3) {min_ext.x - padding, min_ext.y - padding, min_ext.z - padding};
  hi = (Vec3) {max_ext.x + padding, max_ext.y + padding, max_ext.z + padding};
  printf("domain lengths are %g by %g by %g\n", hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);

  lattice_dim = lattice_from_bounding_box(lo, hi, h);
  lattice = create_lattice(lattice_dim);
  printf("\n");

  //Broadcast Atoms
  int numAtoms = atom->size;
  Atom * atom_array = atom->atoms;
  MPI_Bcast(&numAtoms, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(atom_array, numAtoms*sizeof(Atom)/sizeof(float), MPI_FLOAT, 0, MPI_COMM_WORLD);

  //Broadcast Lattice Dimension
  MPI_Bcast(&lattice_dim, sizeof(LatticeDim)/sizeof(float), MPI_FLOAT, 0, MPI_COMM_WORLD);

  //Broadcast Cutoff
  MPI_Bcast(&cutoff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  //Broadcast Cells
  int ncell, nxcell, nycell, nzcell;
  Vec3 minext, maxext;
  get_atom_extent(&minext, &maxext, atom);
  nxcell = (int) floorf((maxext.x-minext.x) * 1.f/4.f) + 1;
  nycell = (int) floorf((maxext.y-minext.y) * 1.f/4.f) + 1;
  nzcell = (int) floorf((maxext.z-minext.z) * 1.f/4.f) + 1;
  ncell = nxcell * nycell * nzcell;
  MPI_Bcast(&nxcell, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nycell, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nzcell, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //Task Partition
  int workSize = (ncell + numProcs - 1) / numProcs;
  int start = workSize * id;
  int end = start + workSize > ncell ? ncell : start + workSize;
  int lattice_size = ((lattice_dim.nx * lattice_dim.ny * lattice_dim.nz) + 7) & ~7;
  float *remote_lattice = (float *)calloc(lattice_size, sizeof(float));

  // MPI kernel
  int i;
  if (mpi_compute_cutoff_potential_lattice(start, end, lattice_dim, remote_lattice, cutoff, atom_array, numAtoms, nxcell, nycell, nzcell)) {
    fprintf(stderr, "Computation failed\n");
    exit(1);
  }

  //Gather Results
  float* local_lattice = lattice->lattice;
  MPI_Reduce(remote_lattice, local_lattice, lattice_size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);


  if (remove_exclusions(lattice, exclcutoff, atom)) {
    fprintf(stderr, "remove_exclusions() failed for cpu lattice\n");
    exit(1);
  }

  // Print output
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  if (parameters->outFile) {
    write_lattice_summary(parameters->outFile, lattice);
  }
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  // Cleanup
  destroy_lattice(lattice);
  free_atom(atom);
  free(remote_lattice);

  MPI_Finalize();

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;
}
