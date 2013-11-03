#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cula.h>
#include <math.h>

#define I(i, j, ld) j * ld + i

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 

float rnorm(){
  float r1 = ((float) rand()) / ((float) RAND_MAX);
  float r2 = ((float) rand()) / ((float) RAND_MAX);
  return sqrt( -2 * log(r1) ) * cos(2 * 3.1415 * r2);
}

int main(){
  int i, j;
  int n = 10;
  int p = 3;
  int* ipiv;
  float k;
  float *X, *XtX, *XtY, *beta, *Y, *dX, *dXtX, *dXtY, *dbeta, *dY;
  
  float *a, *b;
  a = (float*) malloc(sizeof(*X));
  b = (float*) malloc(sizeof(*X));
  *a = 1.0;
  *b = 0.0;
  
  cublasHandle_t handle;
  cublasCreate(&handle);
   
  X = (float*) malloc(n * p * sizeof(*X));
  XtX = (float*) malloc(p * p * sizeof(*X));
  XtY = (float*) malloc(p * sizeof(*X));
  beta = (float*) malloc(p * sizeof(*X));
  Y = (float*) malloc(n * sizeof(*X));
  
  CUDA_CALL(cudaMalloc((void**) &ipiv, p * p * sizeof(*ipiv)));
  CUDA_CALL(cudaMalloc((void**) &dX, n * p * sizeof(*X)));
  CUDA_CALL(cudaMalloc((void**) &dXtX, p * p * sizeof(*X)));
  CUDA_CALL(cudaMalloc((void**) &dXtY, p * sizeof(*X)));
  CUDA_CALL(cudaMalloc((void**) &dbeta, p * sizeof(*X)));
  CUDA_CALL(cudaMalloc((void**) &dY, n * sizeof(*X)));

  printf("Y\t\tX\n");
  for(i = 0; i < n; i++){
    k = (float) i;
    X[I(i, 0, n)] = 1.0;
    X[I(i, 1, n)] = k / 10.0;
    X[I(i, 2, n)] = k * k / 10.0;  
    Y[i] = (k - 5.0) * (k - 2.3) / 3.0 + rnorm();
    
    printf("%0.2f\t\t", Y[i]);
    for(j = 0; j < p; j++){
      printf("%0.2f\t", X[I(i, j, n)]);
    } 
    printf("\n");
  }
  printf("\n");
  
  CUDA_CALL(cudaMemcpy(dX, X, n * p * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dY, Y, n * sizeof(float), cudaMemcpyHostToDevice));

  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, 
    a, dX, n, dX, n, b, dXtX, p);

  CUDA_CALL(cudaMemcpy(XtX, dXtX, p * p * sizeof(float), cudaMemcpyDeviceToHost));

  printf("XtX\n");
  for(i = 0; i < p; i++){
    for(j = 0; j < p; j++){
      printf("%0.2f\t", XtX[I(i, j, p)]);
    }
    printf("\n");
  }
  printf("\n");

  culaInitialize();
  
  culaDeviceSgetrf(p, p, dXtX, p, ipiv);
  culaDeviceSgetri(p, dXtX, p, ipiv);
  
  CUDA_CALL(cudaMemcpy(XtX, dXtX, p * p * sizeof(float), cudaMemcpyDeviceToHost));

  printf("XtX^(-1)\n");
  for(i = 0; i < p; i++){
    for(j = 0; j < p; j++){
      printf("%0.2f\t", XtX[I(i, j, p)]);
    }
    printf("\n");
  }
  printf("\n");
  
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, 1, n, 
    a, dX, n, dY, n, b, dXtY, p);

  cublasSgemv(handle, CUBLAS_OP_N, p, p, 
    a, dXtX, p, dXtY, 1, b, dbeta, 1);

  CUDA_CALL(cudaMemcpy(beta, dbeta, p * sizeof(float), cudaMemcpyDeviceToHost));

  printf("CUBLAS/CULA matrix algebra parameter estimates:\n");
  for(i = 0; i < p; i++){
    printf("beta_%i = %0.2f\n", i, beta[i]);
  }
  printf("\n");


  culaSgels('N', n, p, 1, X, n, Y, n);

  printf("culaSgels Parameter estimates:\n");
  for(i = 0; i < p; i++){
    printf("beta_%i = %0.2f\n", i, Y[i]);
  }
  printf("\n");
  

  culaShutdown();
  cublasDestroy(handle);

  free(a);
  free(b);
  free(X);
  free(XtX);
  free(XtY);
  free(beta);
  free(Y);
  
  CUDA_CALL(cudaFree(dX));
  CUDA_CALL(cudaFree(dXtX));
  CUDA_CALL(cudaFree(dXtY));
  CUDA_CALL(cudaFree(dbeta));
  CUDA_CALL(cudaFree(dY));
}