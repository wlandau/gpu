#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h> 

/*
 * This program computes the sum of the elements of 
 * vector v using the pairwise (cascading) sum algorithm.
 */

#define N 1024 // length of vector v. MUST BE A POWER OF 2!!!

// Fill the vector v with n random floating point numbers.
void vfill(float* v, int n){
  int i;
  for(i = 0; i < n; i++){
    v[i] = (float) rand() / RAND_MAX;
  }
}

// Print the vector v.
void vprint(float* v, int n){
  int i;
  printf("v = \n");
  for(i = 0; i < n; i++){
    printf("%7.3f\n", v[i]);
  }
  printf("\n");
}

// Pairwise-sum the elements of vector v and store the result in v[0]. 
__global__ void psum(float *v){ 
  int t = threadIdx.x; // Thread index.
  int n = blockDim.x; // Should be half the length of v.

  while (n != 0) {
    if(t < n)
      v[t] += v[t + n];  
    __syncthreads();    
    n /= 2; 
  }
}

// Linear sum the elements of vector v and return the result
float lsum(float *v, int len){
  float s = 0;
  int i;
  for(i = 0; i < len; i++){
    s += v[i];
  }
  return s;
}


int main (void){ 
  float *v_h, *v_d; // host and device copies of our vector, respectively
  
  // dynamically allocate memory on the host for v_h
  v_h = (float*) malloc(N * sizeof(*v_h)); 
  
  // dynamically allocate memory on the device for v_d
  cudaMalloc ((float**) &v_d, N *sizeof(*v_d)); 
  
  // Fill v_h with N random floating point numbers.
  vfill(v_h, N);
  
  // Print v_h to the console
  // vprint(v_h, N);
  
  // Write the contents of v_h to v_d
  cudaMemcpy( v_d, v_h, N * sizeof(float), cudaMemcpyHostToDevice );
    
  // compute the linear sum of the elements of v_h on the CPU and return the result
  // also, time the result.
  clock_t start = clock();
  float s = lsum(v_h, N);
  
  float elapsedTime = ((float) clock() - start) / CLOCKS_PER_SEC;
  printf("Linear Sum = %7.3f, CPU Time elapsed: %f seconds\n", s, elapsedTime);
 
  // Compute the pairwise sum of the elements of v_d and store the result in v_d[0].
  // Also, time the computation.
  
  float   gpuElapsedTime;
  cudaEvent_t gpuStart, gpuStop;
  cudaEventCreate(&gpuStart);
  cudaEventCreate(&gpuStop);
  cudaEventRecord( gpuStart, 0 );

  psum<<< 1, N/2 >>>(v_d);
  
  cudaEventRecord( gpuStop, 0 );
  cudaEventSynchronize( gpuStop );
  cudaEventElapsedTime( &gpuElapsedTime, gpuStart, gpuStop ); // time in milliseconds
  cudaEventDestroy( gpuStart );
  cudaEventDestroy( gpuStop );
  
  // Write the pairwise sum, v_d[0], to v_h[0].
  cudaMemcpy(v_h, v_d, sizeof(float), cudaMemcpyDeviceToHost );
  
  // Print the pairwise sum.
  printf("Pairwise Sum = %7.3f, GPU Time elapsed: %f seconds\n", v_h[0], gpuElapsedTime/1000.0);
   
  // Free dynamically-allocated host memory
  free(v_h);

  // Free dynamically-allocated device memory    
  cudaFree(v_d);
}