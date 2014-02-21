#define _CRT_SECURE_NO_WARNINGS
#include "lbm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


#if !defined(SPEC_CPU)
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

/*############################################################################*/

#define DFL1 (1.0f/3.0f)
#define DFL2 (1.0f/18.0f)
#define DFL3 (1.0f/36.0f)

/*############################################################################*/

void LBM_allocateGrid( float** ptr ) {
	const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES,
	             size   = sizeof( LBM_Grid ) + 2*margin*sizeof( float );

	*ptr = (float*)malloc( size );
	if( ! *ptr ) {
		printf( "LBM_allocateGrid: could not allocate %.1f MByte\n",
		        size / (1024.0*1024.0) );
		exit( 1 );
	}
#if !defined(SPEC_CPU)
	printf( "LBM_allocateGrid: allocated %.1f MByte\n",
	        size / (1024.0*1024.0) );
#endif
	*ptr += margin;
}

/*############################################################################*/

void LBM_freeGrid( float** ptr ) {
	const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES;

	free( *ptr-margin );
	*ptr = NULL;
}

void LBM_initializeGrid( LBM_Grid grid ) {
	SWEEP_VAR

	/*voption indep*/
#if !defined(SPEC_CPU)
#ifdef _OPENMP
#pragma omp parallel for
#endif
#endif
	SWEEP_START( 0, 0, -2, 0, 0, SIZE_Z+2 )
		LOCAL( grid, C  ) = DFL1;
		LOCAL( grid, N  ) = DFL2;
		LOCAL( grid, S  ) = DFL2;
		LOCAL( grid, E  ) = DFL2;
		LOCAL( grid, W  ) = DFL2;
		LOCAL( grid, T  ) = DFL2;
		LOCAL( grid, B  ) = DFL2;
		LOCAL( grid, NE ) = DFL3;
		LOCAL( grid, NW ) = DFL3;
		LOCAL( grid, SE ) = DFL3;
		LOCAL( grid, SW ) = DFL3;
		LOCAL( grid, NT ) = DFL3;
		LOCAL( grid, NB ) = DFL3;
		LOCAL( grid, ST ) = DFL3;
		LOCAL( grid, SB ) = DFL3;
		LOCAL( grid, ET ) = DFL3;
		LOCAL( grid, EB ) = DFL3;
		LOCAL( grid, WT ) = DFL3;
		LOCAL( grid, WB ) = DFL3;

		CLEAR_ALL_FLAGS_SWEEP( grid );
	SWEEP_END
}

void LBM_swapGrids( LBM_GridPtr* grid1, LBM_GridPtr* grid2 ) {
	LBM_GridPtr aux = *grid1;
	*grid1 = *grid2;
	*grid2 = aux;
}

void LBM_loadObstacleFile( LBM_Grid grid, const char* filename ) {
	int x,  y,  z;
	FILE* file = fopen( filename, "rb" );
	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( fgetc( file ) != '.' ) SET_FLAG( grid, x, y, z, OBSTACLE );
			}
			fgetc( file );
		}
		fgetc( file );
	}
	fclose( file );
}

void LBM_initializeSpecialCellsForLDC( LBM_Grid grid ) {
	int x,  y,  z;
	for( z = -2; z < SIZE_Z+2; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( x == 0 || x == SIZE_X-1 ||
				    y == 0 || y == SIZE_Y-1 ||
				    z == 0 || z == SIZE_Z-1 ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
				else {
					if( (z == 1 || z == SIZE_Z-2) &&
					     x > 1 && x < SIZE_X-2 &&
					     y > 1 && y < SIZE_Y-2 ) {
						SET_FLAG( grid, x, y, z, myACCEL );
					}
				}
			}
		}
	}
}

void LBM_initializeSpecialCellsForChannel( LBM_Grid grid ) {
	int x,  y,  z;
	for( z = -2; z < SIZE_Z+2; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( x == 0 || x == SIZE_X-1 ||
				    y == 0 || y == SIZE_Y-1 ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );

					if( (z == 0 || z == SIZE_Z-1) &&
					    ! TEST_FLAG( grid, x, y, z, OBSTACLE ))
						SET_FLAG( grid, x, y, z, IN_OUT_FLOW );
				}
			}
		}
	}
}

void LBM_dataLayoutTransform(LBM_Grid old_grid, LBM_Grid new_grid) {
	int x,y,z;
	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				GRID_ENTRY(new_grid, x, y, z, C) = GRID_ENTRY(old_grid, x, y, z, C);
				GRID_ENTRY(new_grid, x, y+1, z, S) = GRID_ENTRY(old_grid, x, y, z, S);
				GRID_ENTRY(new_grid, x, y-1, z, N) = GRID_ENTRY(old_grid, x, y, z, N);
				GRID_ENTRY(new_grid, x+1, y, z, W) = GRID_ENTRY(old_grid, x, y, z, W);
				GRID_ENTRY(new_grid, x-1, y, z, E) = GRID_ENTRY(old_grid, x, y, z, E);
				GRID_ENTRY(new_grid, x, y, z+1, B) = GRID_ENTRY(old_grid, x, y, z, B);
				GRID_ENTRY(new_grid, x, y, z-1, T) = GRID_ENTRY(old_grid, x, y, z, T);
				GRID_ENTRY(new_grid, x+1, y+1, z, SW) = GRID_ENTRY(old_grid, x, y, z, SW);
				GRID_ENTRY(new_grid, x-1, y+1, z, SE) = GRID_ENTRY(old_grid, x, y, z, SE);
				GRID_ENTRY(new_grid, x+1, y-1, z, NW) = GRID_ENTRY(old_grid, x, y, z, NW);
				GRID_ENTRY(new_grid, x-1, y-1, z, NE) = GRID_ENTRY(old_grid, x, y, z, NE);
				GRID_ENTRY(new_grid, x, y+1, z+1, SB) = GRID_ENTRY(old_grid, x, y, z, SB);
				GRID_ENTRY(new_grid, x, y+1, z-1, ST) = GRID_ENTRY(old_grid, x, y, z, ST);
				GRID_ENTRY(new_grid, x, y-1, z+1, NB) = GRID_ENTRY(old_grid, x, y, z, NB);
				GRID_ENTRY(new_grid, x, y-1, z-1, NT) = GRID_ENTRY(old_grid, x, y, z, NT);
				GRID_ENTRY(new_grid, x+1, y, z+1, WB) = GRID_ENTRY(old_grid, x, y, z, WB);
				GRID_ENTRY(new_grid, x+1, y, z-1, WT) = GRID_ENTRY(old_grid, x, y, z, WT);
				GRID_ENTRY(new_grid, x-1, y, z+1, EB) = GRID_ENTRY(old_grid, x, y, z, EB);
				GRID_ENTRY(new_grid, x-1, y, z-1, ET) = GRID_ENTRY(old_grid, x, y, z, ET);
				GRID_ENTRY(new_grid, x, y, z, FLAGS) = GRID_ENTRY(old_grid, x, y, z, FLAGS);
			}
		}
	}
}

void LBM_dataLayoutTransformBack(LBM_Grid from_grid, LBM_Grid to_grid) {
	int x,y,z;
	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				GRID_ENTRY(to_grid, x, y, z, C) = GRID_ENTRY(from_grid, x, y, z, C);
				GRID_ENTRY(to_grid, x, y, z, S) = GRID_ENTRY(from_grid, x, y+1, z, S);
				GRID_ENTRY(to_grid, x, y, z, N) = GRID_ENTRY(from_grid, x, y-1, z, N);
				GRID_ENTRY(to_grid, x, y, z, W) = GRID_ENTRY(from_grid, x+1, y, z, W);
				GRID_ENTRY(to_grid, x, y, z, E) = GRID_ENTRY(from_grid, x-1, y, z, E);
				GRID_ENTRY(to_grid, x, y, z, B) = GRID_ENTRY(from_grid, x, y, z+1, B);
				GRID_ENTRY(to_grid, x, y, z, T) = GRID_ENTRY(from_grid, x, y, z-1, T);
				GRID_ENTRY(to_grid, x, y, z, SW) = GRID_ENTRY(from_grid, x+1, y+1, z, SW);
				GRID_ENTRY(to_grid, x, y, z, SE) = GRID_ENTRY(from_grid, x-1, y+1, z, SE);
				GRID_ENTRY(to_grid, x, y, z, NW) = GRID_ENTRY(from_grid, x+1, y-1, z, NW);
				GRID_ENTRY(to_grid, x, y, z, NE) = GRID_ENTRY(from_grid, x-1, y-1, z, NE);
				GRID_ENTRY(to_grid, x, y, z, SB) = GRID_ENTRY(from_grid, x, y+1, z+1, SB);
				GRID_ENTRY(to_grid, x, y, z, ST) = GRID_ENTRY(from_grid, x, y+1, z-1, ST);
				GRID_ENTRY(to_grid, x, y, z, NB) = GRID_ENTRY(from_grid, x, y-1, z+1, NB);
				GRID_ENTRY(to_grid, x, y, z, NT) = GRID_ENTRY(from_grid, x, y-1, z-1, NT);
				GRID_ENTRY(to_grid, x, y, z, WB) = GRID_ENTRY(from_grid, x+1, y, z+1, WB);
				GRID_ENTRY(to_grid, x, y, z, WT) = GRID_ENTRY(from_grid, x+1, y, z-1, WT);
				GRID_ENTRY(to_grid, x, y, z, EB) = GRID_ENTRY(from_grid, x-1, y, z+1, EB);
				GRID_ENTRY(to_grid, x, y, z, ET) = GRID_ENTRY(from_grid, x-1, y, z-1, ET);
				GRID_ENTRY(to_grid, x, y, z, FLAGS) = GRID_ENTRY(from_grid, x, y, z, FLAGS);
			}
		}
	}
}

void LBM_performStreamCollide( LBM_Grid srcGrid, LBM_Grid dstGrid ) {
	SWEEP_VAR
	float ux, uy, uz, u2, rho;
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			TRANS_DST_C ( dstGrid ) = TRANS_SRC_C ( srcGrid );
			TRANS_DST_S ( dstGrid ) = TRANS_SRC_N ( srcGrid );
			TRANS_DST_N ( dstGrid ) = TRANS_SRC_S ( srcGrid );
			TRANS_DST_W ( dstGrid ) = TRANS_SRC_E ( srcGrid );
			TRANS_DST_E ( dstGrid ) = TRANS_SRC_W ( srcGrid );
			TRANS_DST_B ( dstGrid ) = TRANS_SRC_T ( srcGrid );
			TRANS_DST_T ( dstGrid ) = TRANS_SRC_B ( srcGrid );
			TRANS_DST_SW( dstGrid ) = TRANS_SRC_NE( srcGrid );
			TRANS_DST_SE( dstGrid ) = TRANS_SRC_NW( srcGrid );
			TRANS_DST_NW( dstGrid ) = TRANS_SRC_SE( srcGrid );
			TRANS_DST_NE( dstGrid ) = TRANS_SRC_SW( srcGrid );
			TRANS_DST_SB( dstGrid ) = TRANS_SRC_NT( srcGrid );
			TRANS_DST_ST( dstGrid ) = TRANS_SRC_NB( srcGrid );
			TRANS_DST_NB( dstGrid ) = TRANS_SRC_ST( srcGrid );
			TRANS_DST_NT( dstGrid ) = TRANS_SRC_SB( srcGrid );
			TRANS_DST_WB( dstGrid ) = TRANS_SRC_ET( srcGrid );
			TRANS_DST_WT( dstGrid ) = TRANS_SRC_EB( srcGrid );
			TRANS_DST_EB( dstGrid ) = TRANS_SRC_WT( srcGrid );
			TRANS_DST_ET( dstGrid ) = TRANS_SRC_WB( srcGrid );
			continue;
		}

		rho = + TRANS_SRC_C ( srcGrid ) + TRANS_SRC_N ( srcGrid )
		      + TRANS_SRC_S ( srcGrid ) + TRANS_SRC_E ( srcGrid )
		      + TRANS_SRC_W ( srcGrid ) + TRANS_SRC_T ( srcGrid )
		      + TRANS_SRC_B ( srcGrid ) + TRANS_SRC_NE( srcGrid )
		      + TRANS_SRC_NW( srcGrid ) + TRANS_SRC_SE( srcGrid )
		      + TRANS_SRC_SW( srcGrid ) + TRANS_SRC_NT( srcGrid )
		      + TRANS_SRC_NB( srcGrid ) + TRANS_SRC_ST( srcGrid )
		      + TRANS_SRC_SB( srcGrid ) + TRANS_SRC_ET( srcGrid )
		      + TRANS_SRC_EB( srcGrid ) + TRANS_SRC_WT( srcGrid )
		      + TRANS_SRC_WB( srcGrid );

		ux = + TRANS_SRC_E ( srcGrid ) - TRANS_SRC_W ( srcGrid )
		     + TRANS_SRC_NE( srcGrid ) - TRANS_SRC_NW( srcGrid )
		     + TRANS_SRC_SE( srcGrid ) - TRANS_SRC_SW( srcGrid )
		     + TRANS_SRC_ET( srcGrid ) + TRANS_SRC_EB( srcGrid )
		     - TRANS_SRC_WT( srcGrid ) - TRANS_SRC_WB( srcGrid );
		uy = + TRANS_SRC_N ( srcGrid ) - TRANS_SRC_S ( srcGrid )
		     + TRANS_SRC_NE( srcGrid ) + TRANS_SRC_NW( srcGrid )
		     - TRANS_SRC_SE( srcGrid ) - TRANS_SRC_SW( srcGrid )
		     + TRANS_SRC_NT( srcGrid ) + TRANS_SRC_NB( srcGrid )
		     - TRANS_SRC_ST( srcGrid ) - TRANS_SRC_SB( srcGrid );
		uz = + TRANS_SRC_T ( srcGrid ) - TRANS_SRC_B ( srcGrid )
		     + TRANS_SRC_NT( srcGrid ) - TRANS_SRC_NB( srcGrid )
		     + TRANS_SRC_ST( srcGrid ) - TRANS_SRC_SB( srcGrid )
		     + TRANS_SRC_ET( srcGrid ) - TRANS_SRC_EB( srcGrid )
		     + TRANS_SRC_WT( srcGrid ) - TRANS_SRC_WB( srcGrid );

		ux /= rho;
		uy /= rho;
		uz /= rho;

		if( TEST_FLAG_SWEEP( srcGrid, myACCEL )) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		TRANS_DST_C ( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_C ( srcGrid ) + DFL1*OMEGA*rho*(1.0f - u2);
		TRANS_DST_N ( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_N ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		TRANS_DST_S ( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_S ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		TRANS_DST_E ( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_E ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		TRANS_DST_W ( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_W ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		TRANS_DST_T ( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_T ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		TRANS_DST_B ( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_B ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);
		TRANS_DST_NE( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_NE( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		TRANS_DST_NW( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_NW( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		TRANS_DST_SE( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_SE( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		TRANS_DST_SW( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_SW( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		TRANS_DST_NT( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_NT( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		TRANS_DST_NB( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_NB( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		TRANS_DST_ST( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_ST( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		TRANS_DST_SB( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_SB( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		TRANS_DST_ET( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_ET( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		TRANS_DST_EB( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_EB( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		TRANS_DST_WT( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_WT( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		TRANS_DST_WB( dstGrid ) = (1.0f-OMEGA)*TRANS_SRC_WB( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
	SWEEP_END
}

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	float ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	/* inflow */
	/*voption indep*/
#if !defined(SPEC_CPU)
#ifdef _OPENMP
#pragma omp parallel for private( ux, uy, uz, rho, ux1, uy1, uz1, rho1, \
                                  ux2, uy2, uz2, rho2, u2, px, py )
#endif
#endif
	SWEEP_START( 0, 0, 0, 0, 0, 1 )
		rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );
		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

		rho = 2.0*rho1 - rho2;

		px = (SWEEP_X / (0.5*(SIZE_X-1))) - 1.0;
		py = (SWEEP_Y / (0.5*(SIZE_Y-1))) - 1.0;
		ux = 0.00;
		uy = 0.00;
		uz = 0.01 * (1.0-px*px) * (1.0-py*py);

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0                                 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
	SWEEP_END

	/* outflow */
	/*voption indep*/
#if !defined(SPEC_CPU)
#ifdef _OPENMP
#pragma omp parallel for private( ux, uy, uz, rho, ux1, uy1, uz1, rho1, \
                                  ux2, uy2, uz2, rho2, u2, px, py )
#endif
#endif

	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
		rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		ux1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		uy1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );
		uz1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		ux1 /= rho1;
		uy1 /= rho1;
		uz1 /= rho1;

		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		ux2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		uy2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );
		uz2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		ux2 /= rho2;
		uy2 /= rho2;
		uz2 /= rho2;

		rho = 1.0;

		ux = 2*ux1 - ux2;
		uy = 2*uy1 - uy2;
		uz = 2*uz1 - uz2;

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0                                 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
	SWEEP_END
}

void LBM_showGridStatistics( LBM_Grid grid ) {
	int nObstacleCells = 0,
	    nAccelCells    = 0,
	    nFluidCells    = 0;
	float ux, uy, uz;
	float minU2  = 1e+30, maxU2  = -1e+30, u2;
	float minRho = 1e+30, maxRho = -1e+30, rho;
	float mass = 0;

	SWEEP_VAR

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		rho = + LOCAL( grid, C  ) + LOCAL( grid, N  )
		      + LOCAL( grid, S  ) + LOCAL( grid, E  )
		      + LOCAL( grid, W  ) + LOCAL( grid, T  )
		      + LOCAL( grid, B  ) + LOCAL( grid, NE )
		      + LOCAL( grid, NW ) + LOCAL( grid, SE )
		      + LOCAL( grid, SW ) + LOCAL( grid, NT )
		      + LOCAL( grid, NB ) + LOCAL( grid, ST )
		      + LOCAL( grid, SB ) + LOCAL( grid, ET )
		      + LOCAL( grid, EB ) + LOCAL( grid, WT )
		      + LOCAL( grid, WB );
		if( rho < minRho ) minRho = rho;
		if( rho > maxRho ) maxRho = rho;
		mass += rho;

		if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
			nObstacleCells++;
		}
		else {
			if( TEST_FLAG_SWEEP( grid, myACCEL ))
				nAccelCells++;
			else
				nFluidCells++;

			ux = + LOCAL( grid, E  ) - LOCAL( grid, W  )
			     + LOCAL( grid, NE ) - LOCAL( grid, NW )
			     + LOCAL( grid, SE ) - LOCAL( grid, SW )
			     + LOCAL( grid, ET ) + LOCAL( grid, EB )
			     - LOCAL( grid, WT ) - LOCAL( grid, WB );
			uy = + LOCAL( grid, N  ) - LOCAL( grid, S  )
			     + LOCAL( grid, NE ) + LOCAL( grid, NW )
			     - LOCAL( grid, SE ) - LOCAL( grid, SW )
			     + LOCAL( grid, NT ) + LOCAL( grid, NB )
			     - LOCAL( grid, ST ) - LOCAL( grid, SB );
			uz = + LOCAL( grid, T  ) - LOCAL( grid, B  )
			     + LOCAL( grid, NT ) - LOCAL( grid, NB )
			     + LOCAL( grid, ST ) - LOCAL( grid, SB )
			     + LOCAL( grid, ET ) - LOCAL( grid, EB )
			     + LOCAL( grid, WT ) - LOCAL( grid, WB );
			u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
			if( u2 < minU2 ) minU2 = u2;
			if( u2 > maxU2 ) maxU2 = u2;
		}
	SWEEP_END

        printf( "LBM_showGridStatistics:\n"
        "\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
        "\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
        "\tminU: %e maxU: %e\n\n",
        nObstacleCells, nAccelCells, nFluidCells,
        minRho, maxRho, mass,
        sqrt( minU2 ), sqrt( maxU2 ) );

}

static void storeValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		const char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		int i;

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			buffer[i] = vPtr[sizeof( OUTPUT_PRECISION ) - i - 1];

		fwrite( buffer, sizeof( OUTPUT_PRECISION ), 1, file );
	}
	else {                                                     /* little endian */
		fwrite( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

static void loadValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
		int i;

		fread( buffer, sizeof( OUTPUT_PRECISION ), 1, file );

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			vPtr[i] = buffer[sizeof( OUTPUT_PRECISION ) - i - 1];
	}
	else {                                                     /* little endian */
		fread( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

void LBM_storeVelocityField( LBM_Grid grid, const char* filename,
                             const int binary ) {
	int x, y, z;
	OUTPUT_PRECISION rho, ux, uy, uz;

	FILE* file = fopen( filename, (binary ? "wb" : "w") );

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				rho = + GRID_ENTRY( grid, x, y, z, C  ) + GRID_ENTRY( grid, x, y, z, N  )
				      + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
				      + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
				      + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
				      + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
				      + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
				      + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
				      + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
				      + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
				      + GRID_ENTRY( grid, x, y, z, WB );
				ux = + GRID_ENTRY( grid, x, y, z, E  ) - GRID_ENTRY( grid, x, y, z, W  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW ) 
				     + GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB ) 
				     - GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				uy = + GRID_ENTRY( grid, x, y, z, N  ) - GRID_ENTRY( grid, x, y, z, S  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW ) 
				     - GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB ) 
				     - GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
				uz = + GRID_ENTRY( grid, x, y, z, T  ) - GRID_ENTRY( grid, x, y, z, B  ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB ) 
				     + GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB ) 
				     + GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					/*
					fwrite( &ux, sizeof( ux ), 1, file );
					fwrite( &uy, sizeof( uy ), 1, file );
					fwrite( &uz, sizeof( uz ), 1, file );
					*/
					storeValue( file, &ux );
					storeValue( file, &uy );
					storeValue( file, &uz );
				} else
					fprintf( file, "%e %e %e\n", ux, uy, uz );

			}
		}
	}

	fclose( file );
}

/*############################################################################*/

void LBM_compareVelocityField( LBM_Grid grid, const char* filename,
                             const int binary ) {
	int x, y, z;
	float rho, ux, uy, uz;
	OUTPUT_PRECISION fileUx, fileUy, fileUz,
	                 dUx, dUy, dUz,
	                 diff2, maxDiff2 = -1e+30;

	FILE* file = fopen( filename, (binary ? "rb" : "r") );

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				rho = + GRID_ENTRY( grid, x, y, z, C  ) + GRID_ENTRY( grid, x, y, z, N  )
				      + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
				      + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
				      + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
				      + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
				      + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
				      + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
				      + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
				      + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
				      + GRID_ENTRY( grid, x, y, z, WB );
				ux = + GRID_ENTRY( grid, x, y, z, E  ) - GRID_ENTRY( grid, x, y, z, W  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW ) 
				     + GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB ) 
				     - GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				uy = + GRID_ENTRY( grid, x, y, z, N  ) - GRID_ENTRY( grid, x, y, z, S  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW ) 
				     - GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB ) 
				     - GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
				uz = + GRID_ENTRY( grid, x, y, z, T  ) - GRID_ENTRY( grid, x, y, z, B  ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB ) 
				     + GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB ) 
				     + GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					loadValue( file, &fileUx );
					loadValue( file, &fileUy );
					loadValue( file, &fileUz );
				}
				else {
					if( sizeof( OUTPUT_PRECISION ) == sizeof( double )) {
						fscanf( file, "%lf %lf %lf\n", &fileUx, &fileUy, &fileUz );
					}
					else {
						fscanf( file, "%f %f %f\n", &fileUx, &fileUy, &fileUz );
					}
				}

				dUx = ux - fileUx;
				dUy = uy - fileUy;
				dUz = uz - fileUz;
				diff2 = dUx*dUx + dUy*dUy + dUz*dUz;
				if( diff2 > maxDiff2 ) maxDiff2 = diff2;
			}
		}
	}

#if defined(SPEC_CPU)
	printf( "LBM_compareVelocityField: maxDiff = %e  \n\n",
	        sqrt( maxDiff2 )  );
#else
	printf( "LBM_compareVelocityField: maxDiff = %e  ==>  %s\n\n",
	        sqrt( maxDiff2 ),
	        sqrt( maxDiff2 ) > 1e-5 ? "##### ERROR #####" : "OK" );
#endif
	fclose( file );
}

