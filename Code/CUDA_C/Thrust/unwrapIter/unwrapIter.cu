#include <thrust/device_vector.h>
#include <iostream>
#include <cuda.h>

void __global__ myKernel(int N, int *ptr){
}

int main(){
  int N = 512;
  
  // allocate device vector
  thrust::device_vector<int> d_vec(4);
  
  // obtain raw pointer to device vector's memory
  int *ptr = thrust::raw_pointer_cast(&d_vec[0]);
  
  // use pointer in a CUDA C kernel
  myKernel<<<N/256, 256>>>(N, ptr);
  
  // Note: ptr cannot be dereferenced on the host!
  
  return 0; 
}