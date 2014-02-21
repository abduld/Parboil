# 1 "../opencl_nvidia/kernel.cl"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "../opencl_nvidia/kernel.cl"
# 12 "../opencl_nvidia/kernel.cl"
# 1 "../opencl_nvidia/layout_config.h" 1
# 54 "../opencl_nvidia/layout_config.h"
typedef enum {C = 0,
              N, S, E, W, T, B,
              NE, NW, SE, SW,
              NT, NB, ST, SB,
              ET, EB, WT, WB,
              FLAGS, N_CELL_ENTRIES} CELL_ENTRIES;



typedef enum {OBSTACLE = 1 << 0,
              ACCEL = 1 << 1,
              IN_OUT_FLOW = 1 << 2} CELL_FLAGS;
# 13 "../opencl_nvidia/kernel.cl" 2
# 1 "../opencl_nvidia/lbm_macros.h" 1
# 26 "../opencl_nvidia/lbm_macros.h"
typedef float* LBM_Grid;
typedef LBM_Grid* LBM_GridPtr;
# 14 "../opencl_nvidia/kernel.cl" 2


__kernel void performStreamCollide_kernel( __global float* srcGrid, __global float* dstGrid )
{
 srcGrid += (( (((120)+(8))*((120)+(0))*((150)+(4)))*0 + ((0)+(0)*((120)+(8))+(2)*((120)+(8))*((120)+(0))) ) - ( (((120)+(8))*((120)+(0))*((150)+(4)))*0 + ((0)+(0)*((120)+(8))+(0)*((120)+(8))*((120)+(0))) ));
 dstGrid += (( (((120)+(8))*((120)+(0))*((150)+(4)))*0 + ((0)+(0)*((120)+(8))+(2)*((120)+(8))*((120)+(0))) ) - ( (((120)+(8))*((120)+(0))*((150)+(4)))*0 + ((0)+(0)*((120)+(8))+(0)*((120)+(8))*((120)+(0))) ));





        int __temp_x__, __temp_y__, __temp_z__;
 __temp_x__ = get_local_id(0);
 __temp_y__ = get_group_id(0);
 __temp_z__ = get_group_id(1);

 float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
 float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST ;
 float tempSB, tempET, tempEB, tempWT, tempWB ;




 tempC = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*C + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )])));

 tempN = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*N + (((0)+__temp_x__)+((-1)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempS = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*S + (((0)+__temp_x__)+((+1)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempE = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*E + (((-1)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempW = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*W + (((+1)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempT = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*T + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((-1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempB = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*B + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((+1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));

 tempNE = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*NE + (((-1)+__temp_x__)+((-1)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempNW = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*NW + (((+1)+__temp_x__)+((-1)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempSE = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*SE + (((-1)+__temp_x__)+((+1)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempSW = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*SW + (((+1)+__temp_x__)+((+1)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempNT = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*NT + (((0)+__temp_x__)+((-1)+__temp_y__)*((120)+(8))+((-1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempNB = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*NB + (((0)+__temp_x__)+((-1)+__temp_y__)*((120)+(8))+((+1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempST = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*ST + (((0)+__temp_x__)+((+1)+__temp_y__)*((120)+(8))+((-1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempSB = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*SB + (((0)+__temp_x__)+((+1)+__temp_y__)*((120)+(8))+((+1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempET = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*ET + (((-1)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((-1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempEB = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*EB + (((-1)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((+1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempWT = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*WT + (((+1)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((-1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));
 tempWB = ((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*WB + (((+1)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((+1)+__temp_z__)*((120)+(8))*((120)+(0))) )])));

         float ux, uy, uz, rho, u2;
  float temp1, temp2, temp_base;

 if(as_uint((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*FLAGS + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) & (OBSTACLE)) {



  temp_swp = tempN ; tempN = tempS ; tempS = temp_swp ;
  temp_swp = tempE ; tempE = tempW ; tempW = temp_swp;
  temp_swp = tempT ; tempT = tempB ; tempB = temp_swp;
  temp_swp = tempNE; tempNE = tempSW ; tempSW = temp_swp;
  temp_swp = tempNW; tempNW = tempSE ; tempSE = temp_swp;
  temp_swp = tempNT ; tempNT = tempSB ; tempSB = temp_swp;
  temp_swp = tempNB ; tempNB = tempST ; tempST = temp_swp;
  temp_swp = tempET ; tempET= tempWB ; tempWB = temp_swp;
  temp_swp = tempEB ; tempEB = tempWT ; tempWT = temp_swp;
 }
 else {


  rho = tempC + tempN
   + tempS + tempE
   + tempW + tempT
   + tempB + tempNE
   + tempNW + tempSE
   + tempSW + tempNT
   + tempNB + tempST
   + tempSB + tempET
   + tempEB + tempWT
   + tempWB;

  ux = + tempE - tempW
   + tempNE - tempNW
   + tempSE - tempSW
   + tempET + tempEB
   - tempWT - tempWB;

  uy = + tempN - tempS
   + tempNE + tempNW
   - tempSE - tempSW
   + tempNT + tempNB
   - tempST - tempSB;

  uz = + tempT - tempB
   + tempNT - tempNB
   + tempST - tempSB
   + tempET - tempEB
   + tempWT - tempWB;

  ux /= rho;
  uy /= rho;
  uz /= rho;

  if(as_uint((((srcGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*FLAGS + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) & (ACCEL)) {

   ux = 0.005f;
   uy = 0.002f;
   uz = 0.000f;
  }

  u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
  temp_base = (1.95f)*rho;
  temp1 = (1.0f/ 3.0f)*temp_base;


  temp_base = (1.95f)*rho;
  temp1 = (1.0f/ 3.0f)*temp_base;
  temp2 = 1.0f-(1.95f);
  tempC = temp2*tempC + temp1*( - u2);
         temp1 = (1.0f/18.0f)*temp_base;
  tempN = temp2*tempN + temp1*( uy*(4.5f*uy + 3.0f) - u2);
  tempS = temp2*tempS + temp1*( uy*(4.5f*uy - 3.0f) - u2);
  tempT = temp2*tempT + temp1*( uz*(4.5f*uz + 3.0f) - u2);
  tempB = temp2*tempB + temp1*( uz*(4.5f*uz - 3.0f) - u2);
  tempE = temp2*tempE + temp1*( ux*(4.5f*ux + 3.0f) - u2);
  tempW = temp2*tempW + temp1*( ux*(4.5f*ux - 3.0f) - u2);
  temp1 = (1.0f/36.0f)*temp_base;
  tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
  tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
  tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
  tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
  tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
  tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
  tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
  tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
  tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
  tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
  tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
  tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
 }




 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*C + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempC;

 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*N + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempN;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*S + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempS;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*E + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempE;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*W + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempW;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*T + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempT;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*B + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempB;

 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*NE + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempNE;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*NW + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempNW;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*SE + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempSE;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*SW + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempSW;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*NT + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempNT;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*NB + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempNB;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*ST + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempST;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*SB + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempSB;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*ET + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempET;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*EB + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempEB;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*WT + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempWT;
 ((((dstGrid)[( (((120)+(8))*((120)+(0))*((150)+(4)))*WB + (((0)+__temp_x__)+((0)+__temp_y__)*((120)+(8))+((0)+__temp_z__)*((120)+(8))*((120)+(0))) )]))) = tempWB;
}
