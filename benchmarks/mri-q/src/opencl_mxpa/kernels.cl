#include "macros.h"

__kernel void
ComputePhiMag_GPU(__global float* phiR, __global float* phiI, __global float* phiMag, int numK) {
  int indexK = get_global_id(0);
  if (indexK >= numK)
  	indexK = numK;

  float real, imag;
    real = phiR[indexK];
    imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
}

__kernel void
ComputeQ_GPU(int numK, int kGlobalIndex,
	     __global float* x, __global float* y, __global float* z,
	     __global float* Qr, __global float* Qi, __global struct kValues* ck) 
{
  float sX;
  float sY;
  float sZ;
  float sQr;
  float sQi;

  // Determine the element of the X arrays computed by this thread
  int xIndex = get_global_id(0);

  // Read block's X values from global mem to shared mem
  sX = x[xIndex];
  sY = y[xIndex];
  sZ = z[xIndex];
  sQr = Qr[xIndex];
  sQi = Qi[xIndex];

  int kIndex = 0;
  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex ++, kGlobalIndex ++) {
    float expArg = PIx2 * (ck[kIndex].Kx * sX +
			   ck[kIndex].Ky * sY +
			   ck[kIndex].Kz * sZ);
    sQr = sQr + ck[kIndex].PhiMag * cos(expArg);
    sQi = sQi + ck[kIndex].PhiMag * sin(expArg);
  }

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;
}
