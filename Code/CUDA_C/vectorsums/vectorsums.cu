#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h> 

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 

#define N 10

__global__ void add(int *a, int *b, int *c){
  int bid = blockIdx.x;
  if(bid < N)
    c[bid] = a[bid] + b[bid];
}

int main(void) {
  int i, a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  CUDA_CALL(cudaMalloc((void**) &dev_a, N*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**) &dev_b, N*sizeof(int)));
  CUDA_CALL(cudaMalloc((void**) &dev_c, N*sizeof(int)));

  for(i=0; i<N; i++){
    a[i] = -i;
    b[i] = i*i;
  }

  CUDA_CALL(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

  add<<<N,1>>>(dev_a, dev_b, dev_c);

  CUDA_CALL(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));

  printf("\na + b = c\n");
  for(i = 0; i<N; i++){
    printf("%5d + %5d = %5d\n", a[i], b[i], c[i]);
  }

  CUDA_CALL(cudaFree(dev_a));
  CUDA_CALL(cudaFree(dev_b));
  CUDA_CALL(cudaFree(dev_c));
}