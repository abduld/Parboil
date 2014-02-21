// Xuhao Chen <cxh@illinois.edu>
// University of Illinois
#include "main.h"
#include "lbm.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
static LBM_GridPtr srcGrid, dstGrid;
struct pb_TimerSet timers;
int main( int nArgs, char* arg[] ) {
	printf("Sequencial Lattice Boltzmann Methods by Xuhao Chen\n");
	MAIN_Param param;
	int t;
	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	struct pb_Parameters* params;
	params = pb_ReadParameters(&nArgs, arg);
	MAIN_parseCommandLine( nArgs, arg, &param, params );
	MAIN_printInfo( &param );
	MAIN_initialize( &param );
	LBM_GridPtr trans_src_grid, trans_dst_grid;
	LBM_allocateGrid((float**) &trans_src_grid);
	LBM_allocateGrid((float**) &trans_dst_grid);
	LBM_initializeGrid( *trans_src_grid );
	LBM_initializeGrid( *trans_dst_grid );
	LBM_dataLayoutTransform(*srcGrid, *trans_src_grid);
	LBM_dataLayoutTransform(*dstGrid, *trans_dst_grid);
	for( t = 1; t <= param.nTimeSteps; t++ ) {
		if( param.simType == CHANNEL ) {
			LBM_handleInOutFlow( *srcGrid );
		}
		LBM_performStreamCollide( *trans_src_grid, *trans_dst_grid );
		LBM_swapGrids( &trans_src_grid, &trans_dst_grid );
		if( (t & 63) == 0 ) {
			printf( "timestep: %i\n", t );
		}
	}

	LBM_dataLayoutTransformBack(*trans_src_grid, *srcGrid);
	LBM_dataLayoutTransformBack(*trans_dst_grid, *dstGrid);
	MAIN_finalize( &param );
	LBM_freeGrid( (float**) &trans_src_grid );
	LBM_freeGrid( (float**) &trans_dst_grid );

	pb_SwitchToTimer(&timers, pb_TimerID_NONE);
	pb_PrintTimerSet(&timers);
	pb_FreeParameters(params);
	return 0;
}

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param, struct pb_Parameters * params) {
	struct stat fileStat;
	if( nArgs < 2 ) {
		printf( "syntax: lbm <time steps>\n" );
		exit( 1 );
	}
	param->nTimeSteps = atoi( arg[1] );
	if( params->inpFiles[0] != NULL ) {
		param->obstacleFilename = params->inpFiles[0];
		if( stat( param->obstacleFilename, &fileStat ) != 0 ) {
			printf( "MAIN_parseCommandLine: cannot stat obstacle file '%s'\n",
			         param->obstacleFilename );
			exit( 1 );
		}
		if( fileStat.st_size != SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z ) {
			printf( "MAIN_parseCommandLine:\n"
			        "\tsize of file '%s' is %i bytes\n"
					    "\texpected size is %i bytes\n",
			        param->obstacleFilename, (int) fileStat.st_size,
			        SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z );
			exit( 1 );
		}
	}
	else param->obstacleFilename = NULL;
	param->resultFilename = params->outFile;
	param->action         = STORE;
	param->simType        = LDC;
}

void MAIN_printInfo( const MAIN_Param* param ) {
	const char actionString[3][32] = {"nothing", "compare", "store"};
	const char simTypeString[3][32] = {"lid-driven cavity", "channel flow"};
	printf( "MAIN_printInfo:\n"
	        "\tgrid size      : %i x %i x %i = %.2f * 10^6 Cells\n"
	        "\tnTimeSteps     : %i\n"
	        "\tresult file    : %s\n"
	        "\taction         : %s\n"
	        "\tsimulation type: %s\n"
	        "\tobstacle file  : %s\n\n",
	        SIZE_X, SIZE_Y, SIZE_Z, 1e-6*SIZE_X*SIZE_Y*SIZE_Z,
	        param->nTimeSteps, param->resultFilename, 
	        actionString[param->action], simTypeString[param->simType],
	        (param->obstacleFilename == NULL) ? "<none>" :
	                                            param->obstacleFilename );
}

void MAIN_initialize( const MAIN_Param* param ) {
	LBM_allocateGrid( (float**) &srcGrid );
	LBM_allocateGrid( (float**) &dstGrid );
	LBM_initializeGrid( *srcGrid );
	LBM_initializeGrid( *dstGrid );
	if( param->obstacleFilename != NULL ) {
		LBM_loadObstacleFile( *srcGrid, param->obstacleFilename );
		LBM_loadObstacleFile( *dstGrid, param->obstacleFilename );
	}
	if( param->simType == CHANNEL ) {
		LBM_initializeSpecialCellsForChannel( *srcGrid );
		LBM_initializeSpecialCellsForChannel( *dstGrid );
	}
	else {
		LBM_initializeSpecialCellsForLDC( *srcGrid );
		LBM_initializeSpecialCellsForLDC( *dstGrid );
	}
	LBM_showGridStatistics( *srcGrid );
}

void MAIN_finalize( const MAIN_Param* param ) {
	LBM_showGridStatistics( *srcGrid );
//	printf("result filename: %s\n", param->resultFilename);
	if( param->action == COMPARE )
		LBM_compareVelocityField( *srcGrid, param->resultFilename, TRUE );
	if( param->action == STORE )
	    LBM_storeVelocityField( *srcGrid, param->resultFilename, TRUE );
	LBM_freeGrid( (float**) &srcGrid );
	LBM_freeGrid( (float**) &dstGrid );
}
