/*
* This program uses the host CURAND API to generate 10 pseudorandom floats.
*/

#include <stdio.h> 
#include <stdlib.h>
#include <cuda.h> 
#include <curand.h>

int main(int argc, char *argv[]){
  size_t n = 10;
  size_t i; 
  curandGenerator_t gen; 
  float *devData , *hostData;
  
  /* Allocate n floats on host */
  hostData = (float *) calloc(n, sizeof(float));
  
  /* Allocate n floats on device */ 
  cudaMalloc((void **) &devData, n*sizeof(float));

  /* Create a Mersenne Twister pseudorandom number generator */ 
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);

  /* Set seed */ 
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  
  /* Generate n floats on device */
  curandGenerateUniform(gen, devData, n);
  
  /* Copy device memory to host */ 
  cudaMemcpy(hostData , devData , n * sizeof(float), cudaMemcpyDeviceToHost);

  /* Show result */
  printf("Random Unif(0, 1) draws:\n");
  for(i = 0; i < n; i++) {
    printf("  %1.4f\n", hostData[i]); 
  }
  printf("\n");

  /* Cleanup */ 
  curandDestroyGenerator(gen); 
  cudaFree(devData);
  free(hostData);
}