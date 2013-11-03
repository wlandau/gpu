/*
 * This program uses the device CURAND API to calculate what 
 * proportion of pseudo-random ints are odd.
 */

#include <stdio.h>
#include <stdlib.h> 
#include <cuda.h>
#include <curand_kernel.h>

__global__ void setup_kernel(curandState *state){
  int id = threadIdx.x + blockIdx.x * 64;

  /* Each thread gets same seed, a different sequence number , no offset */
  curand_init(1234, id, 0, &state[id]); 
}
  
__global__ void generate_kernel(curandState *state, int *result){
  int id = threadIdx.x + blockIdx.x * 64; int count = 0;
  unsigned int x;

  /* Copy state to local memory for efficiency */ 
  curandState localState = state[id];
  
  /* Generate pseudo -random unsigned ints */ 
  for(int n = 0; n < 100000; n++){
    x = curand(&localState); 
    
    /* Check if odd */ 
    if(x & 1){
      count ++; 
    }
  }

  /* Copy state back to global memory */ 
  state[id] = localState;

  /* Store results */
  result[id] += count;
}

int main(int argc, char *argv[]){
  int i, total;

  int *devResults, *hostResults;
  curandState *devStates;

  /* Allocate space for results on host */ 
  hostResults = (int *) calloc(64 * 64, sizeof(int));
  
  /* Allocate space for results on device */ 
  cudaMalloc((void **)&devResults , 64 * 64 *sizeof(int));
  
  /* Set results to 0 */ 
  cudaMemset(devResults , 0, 64 * 64 * sizeof(int));
  
  /* Allocate space for prng states on device */ 
  cudaMalloc((void **)&devStates , 64 * 64 * sizeof(curandState)); 
  
  /* Setup prng states */
  setup_kernel<<<64, 64>>>(devStates);
  
  /* Generate and use pseudorandom numbers*/ 
  for(i = 0; i < 10; i++){
    generate_kernel<<<64, 64>>>(devStates, devResults);
  }
  
  /* Copy device memory to host */ 
  cudaMemcpy(hostResults, devResults , 64 * 64 * sizeof(int), cudaMemcpyDeviceToHost);

  /* Show result */
  total = 0;
  for(i = 0; i < 64 * 64; i++) {
    total += hostResults[i];
  }
  printf("Fraction odd was %10.13f\n", (float) total / (64.0f * 64.0f * 100000.0f * 10.0f)); 
  
  /* Cleanup */
  cudaFree(devStates);
  cudaFree(devResults);
  free(hostResults);
  
  return EXIT_SUCCESS;
}