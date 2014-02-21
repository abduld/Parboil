/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * C implementation of MM, adjusted for improved CPU performance.
 */



void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  // Transpose A and B.
  // Transposition allows the dot-product calculation to process contiguous (instead of strided)
  // array elements, improving performance of the main loop.
  float *A2 = (float *)malloc(m * k * sizeof(float));
  float *B2 = (float *)malloc(n * k * sizeof(float));
  {
    for (int i = 0; i < k; i++) {
      for (int mm = 0; mm < m; mm++) {
        A2[mm * k + i] = A[mm + i * lda];
      }
    }
    for (int i = 0; i < k; i++) {
      for (int nn = 0; nn < n; nn++) {
        B2[nn * k + i] = B[nn + i * ldb];
      }
    }
  }

  // Multiply matrices
  {
    for (int mm = 0; mm < m; ++mm) {
      for (int nn = 0; nn < n; ++nn) {
        float c = 0.0f;
        for (int i = 0; i < k; ++i) {
          float a = A2[mm * k + i]; 
          float b = B2[nn * k + i];
          c += a * b;
        }
        C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
      }
    }
  }
  free(A2);
  free(B2);
}
