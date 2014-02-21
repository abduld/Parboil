// (C) Copyright 2013, University of Illinois. All Rights Reserved.
// Xuhao Chen <cxh.nudt@gmail.com>
// MPI implementation

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc ) {
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  for(int mm=0; mm<m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm*k+i];
        float b = B[nn*k+i];
        c += a * b;
      }
      C[mm*n+nn] = C[mm*n+nn] * beta + alpha * c;
    }
  }
}
