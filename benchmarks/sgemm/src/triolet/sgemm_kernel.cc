/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Base C implementation of MM
 */

#include <TrioletData.h>
#include "matmul_cxx.h"

using namespace Triolet;

Array2<Float>
createArray (int height, int width, int stride, const float *data)
{
  int i, j;
  Incomplete<Array2<Float> > make_array;
  make_array.create(0, height, 0, width);
  for (j = 0; j < height; j++)
    for (i = 0; i < width; i++)
      make_array.at(j, i) = (Float)(float)data[j * stride + i];

  return make_array.freeze();
}

void
fromArray (int height, int width, int stride, float *ret, Array2<Float> a)
{
  int i, j;
  for (j = 0; j < height; j++)
    for (i = 0; i < width; i++)
      ret[j * stride + i] = (float)(Float)a.at(j, i);
}

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha,
                 const float *A, int lda, const float *B, int ldb, float beta,
                 float *C, int ldc,
                 struct pb_TimerSet *timers)
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  pb_SwitchToTimer(timers, pb_TimerID_COPY);

  Array2<Float> Ap = createArray(k, m, lda, A);
  Array2<Float> Bp = createArray(k, n, ldb, B);
  Array2<Float> Cp = createArray(n, m, ldc, C);

  pb_SwitchToTimer(timers, pb_TimerID_KERNEL);
  Array2<Float> C_new = sgemm_accel(Ap, Bp, Cp, alpha, beta);

  pb_SwitchToTimer(timers, pb_TimerID_COPY);
  fromArray(n, m, ldc, C, C_new);

  pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
}
