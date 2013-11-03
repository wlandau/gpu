#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h> 

/*
 * This program computes A * B and stores the result as C, where:
 *   A is an M x N matrix (ld = M)
 *   B is an N x P matrix (ld = N)
 *   C is an M x P matrix (ld = M)
 */

#define M 4 
#define N 4 
#define P 4
#define M2A(i, j , ld) i * ld + j 

void mfill(float* A, int nrow, int ncol){
  int i, j;
  for(i = 0; i < nrow; i++){
    for(j = 0; j < ncol; j++){
      A[M2A(i, j, ncol)] =  (float) rand() / RAND_MAX;
    }
  } 
}

void mprint(float* A, int nrow, int ncol){
  int i,j;
  printf("matrix(c(");
  for (j = 0; j < ncol; j++) {
    for (i = 0; i < nrow; i++) {
      if(i == nrow - 1 && j == ncol - 1)
        printf ("%7.3f), ncol = %d)", A[M2A(i, j, ncol)], ncol);
      else  
        printf ("%7.3f,", A[M2A(i, j, ncol)]);
    }
    printf("\n");
  }
}

__global__ void mmul(float* A, float* B, float* C){
  int i = blockIdx.x;  // (i = 1, ..., M)
  int j = blockIdx.y;  // (j = 1, ..., P)
  int k = threadIdx.x; // (k = 1, ..., N)
  __shared__ float c_ij[N];
  
  c_ij[k] = A[M2A(i, k, N)] * B[M2A(k, j, P)];
  
  __syncthreads();
  
  int t = blockDim.x / 2;
  while (t != 0) {
    if(k < t)
      c_ij[k] += c_ij[k + t];
    __syncthreads();
    t /= 2; 
  }
  
  C[M2A(i, j, P)] = c_ij[0];
}

int main (void){ 
  
  float *A_h, *B_h, *C_h; // matrices A, B, and C on the host (CPU) 
  float *A_d, *B_d, *C_d; // matrices A, B, and C on the device (GPU)
  
  // dynamically allocate memory on the host for A, B, and C
  A_h = (float*) malloc(M * N * sizeof(*A_h)); 
  B_h = (float*) malloc(N * P * sizeof(*B_h)); 
  C_h = (float*) malloc(M * P * sizeof(*C_h)); 
  
  // dynamically allocate memory on the device for A, B, and C
  cudaMalloc ((float**) &A_d, M * N *sizeof(*A_h)); 
  cudaMalloc ((float**) &B_d, N * P *sizeof(*B_h));
  cudaMalloc ((float**) &C_d, M * P *sizeof(*C_h));
  
  // Fill A_h and B_h on the host
  mfill(A_h, M, N);
  mfill(B_h, N, P);
  
  // Print A_h and B_h to the console
  printf("A = \n");
  mprint(A_h, M, N);
  printf("B = \n");
  mprint(B_h, N, P);
  
  // Write the contents of A_h and B_h to the device matrices, A_d and B_d
  cudaMemcpy( A_d, A_h, M * N * sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( B_d, B_h, N * P * sizeof(float), cudaMemcpyHostToDevice );
  
  // Multiply matrices A_d and B_d on the device and store result as C_d on the device.
  dim3 grid(M, P);
  mmul<<< grid, N >>>(A_d, B_d, C_d);
  
  // Write the contents of C_d to the host matrix, C_h.
  cudaMemcpy(C_h, C_d, M * P * sizeof(float), cudaMemcpyDeviceToHost );
  
  // print C_h to the console
  printf("C = \n");
  mprint(C_h, M, P);
  
  // Free dynamically-allocated host memory
  free(A_h);
  free(B_h);
  free(C_h);

  // Free dynamically-allocated device memory    
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}