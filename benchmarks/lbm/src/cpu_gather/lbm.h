#ifndef _LBM_H_
#define _LBM_H_
#include "config.h"
#include "lbm_1d_array.h"
#define N_DISTR_FUNCS FLAGS
typedef enum {OBSTACLE = 1 << 0, myACCEL = 1 << 1, IN_OUT_FLOW = 1 << 2} CELL_FLAGS;
void LBM_allocateGrid( float** ptr );
void LBM_freeGrid( float** ptr );
void LBM_initializeGrid( LBM_Grid grid );
void LBM_initializeSpecialCellsForLDC( LBM_Grid grid );
void LBM_loadObstacleFile( LBM_Grid grid, const char* filename );
void LBM_initializeSpecialCellsForChannel( LBM_Grid grid );
void LBM_swapGrids( LBM_GridPtr* grid1, LBM_GridPtr* grid2 );
void LBM_performStreamCollide( LBM_Grid srcGrid, LBM_Grid dstGrid );
void LBM_handleInOutFlow( LBM_Grid srcGrid );
void LBM_showGridStatistics( LBM_Grid Grid );
void LBM_storeVelocityField( LBM_Grid grid, const char* filename, const BOOL binary );
void LBM_compareVelocityField( LBM_Grid grid, const char* filename, const BOOL binary );
void LBM_dataLayoutTransform(LBM_Grid old_grid, LBM_Grid new_grid);
void LBM_dataLayoutTransformBack(LBM_Grid from_grid, LBM_Grid to_grid);
#endif /* _LBM_H_ */
