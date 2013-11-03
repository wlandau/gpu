#include <thrust/fill.h>
#include <cuda.h>
#include <iostream>

int main(){
  int N = 10;

  // raw pointer to device memory
  int* rawptr;
  cudaMalloc((void **) &rawptr, N*sizeof(int));

  //wrap raw pointer with a device ptr
  thrust::device_ptr<int> devptr(rawptr);
  
  //use device ptr in thrust algorithms
  thrust::fill(devptr, devptr + N, (int) 0);
  
  devptr[0] = 1;
  cudaFree(rawptr);
  
  return 0;
  
}