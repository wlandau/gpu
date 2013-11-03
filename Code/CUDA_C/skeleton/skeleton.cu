#include <stdio.h> 
#include <stdlib.h> 
#include <cuda.h>
#include <cuda_runtime.h> 

__global__ void some_kernel(...){...}

int main (void){ 
  // Declare all variables.
  ...
  // Allocate host memory.
  ...
  // Dynamically allocate device memory for GPU results.
  ...
  // Write to host memory.
  ... 
  // Copy host memory to device memory.
  ...
  // Execute kernel on the device.
  some_kernel<<< num_blocks, num_theads_per_block >>>(...);
  
  // Write GPU results in device memory back to host memory.
  ...
  // Free dynamically-allocated host memory
  ...
  // Free dynamically-allocated device memory    
  ...
}