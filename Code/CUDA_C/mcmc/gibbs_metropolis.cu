/*
Created by Zebulun Arendsee.
March 26, 2013

Modified by Will Landau.
June 30, 2013
will-landau.com
landau@iastate.edu

This program implements a MCMC algorithm for the following hierarchical
model:

y_k     ~ Poisson(n_k * theta_k)     k = 1, ..., K
theta_k ~ Gamma(a, b)
a       ~ Unif(0, a0)
b       ~ Unif(0, b0) 

We let a0 and b0 be arbitrarily large.

Arguments:
    1) input filename
        With two space delimited columns holding integer values for
        y and float values for n.
    2) number of trials (1000 by default)

Output: A comma delimited file containing a column for a, b, and each
theta. All output is written to stdout.

Example dataset:

$ head -3 data.txt
4 0.91643
23 3.23709
7 0.40103

Example of compilation and execution:

$ nvcc gibbs_metropolis.cu -o gibbs
$ ./gibbs mydata.txt 2500 > output.csv
$

This code borrows from the nVidia developer zone documentation, 
specifically http://docs.nvidia.com/cuda/curand/index.html#topic_1_2_1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>

#define PI 3.14159265359f
#define THREADS_PER_BLOCK 64

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 

#define CURAND_CALL(x) {if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}}

__host__ void load_data(int argc, char **argv, int *K, int **y, float **n);

__host__ float sample_a(float a, float b, int K, float sum_logs);
__host__ float sample_b(float a, int K, float flat_sum);

__host__ float rnorm();
__host__ float rgamma(float a, float b);

__device__ float rgamma(curandState *state, int id, float a, float b);

__global__ void sample_theta(curandState *state, float *theta, float *log_theta, 
                             int *y, float *n, float a, float b, int K);
__global__ void setup_kernel(curandState *state, unsigned int seed, int);


int main(int argc, char **argv){

  curandState *devStates;
  float a, b, flat_sum, sum_logs, *n, *dev_n, *dev_theta, *dev_log_theta;
  int i, K, *y, *dev_y, nBlocks, trials = 1000;

  if(argc > 2)
    trials = atoi(argv[2]);

  load_data(argc, argv, &K, &y, &n);


  /*------ Allocate memory -----------------------------------------*/

  CUDA_CALL(cudaMalloc((void **)&dev_y, K * sizeof(int)));
  CUDA_CALL(cudaMemcpy(dev_y, y, K * sizeof(int), 
            cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMalloc((void **)&dev_n, K * sizeof(float)));
  CUDA_CALL(cudaMemcpy(dev_n, n, K * sizeof(float), 
            cudaMemcpyHostToDevice));

  /* Allocate space for theta and log_theta on device and host */
  CUDA_CALL(cudaMalloc((void **)&dev_theta, K * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&dev_log_theta, K * sizeof(float)));

  /* Allocate space for random states on device */
  CUDA_CALL(cudaMalloc((void **)&devStates, K * sizeof(curandState)));


  /*------ Setup random number generators (one per thread) ---------*/

  nBlocks = (K + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  setup_kernel<<<nBlocks, THREADS_PER_BLOCK>>>(devStates, 0, K);


  /*------ MCMC ----------------------------------------------------*/
    
  printf("alpha, beta\n");

  /* starting values of hyperparameters */
  a = 20; 
  b = 1; 
    
  /* Steps of MCMC */  
  for(i = 0; i < trials; i++){    
    sample_theta<<<nBlocks, THREADS_PER_BLOCK>>>(devStates, dev_theta, dev_log_theta,
                                                 dev_y, dev_n, a, b, K);

    /* Make iterators for thetas and log thetas. */
    thrust::device_ptr<float> theta(dev_theta);
    thrust::device_ptr<float> log_theta(dev_log_theta);
    
    /* Compute pairwise sums of thetas and log_thetas. */
    flat_sum = thrust::reduce(theta, theta + K);
    sum_logs = thrust::reduce(log_theta, log_theta + K);
  
    /* Sample hyperparameters. */
    a = sample_a(a, b, K, sum_logs);
    b = sample_b(a, K, flat_sum);
    
    /* print hyperparameters. */
    printf("%f, %f\n", a, b); 
  }

  /*------ Free Memory -------------------------------------------*/

  free(y);
  free(n);

  CUDA_CALL(cudaFree(devStates));
  CUDA_CALL(cudaFree(dev_theta));
  CUDA_CALL(cudaFree(dev_log_theta));
  CUDA_CALL(cudaFree(dev_y));
  CUDA_CALL(cudaFree(dev_n));

  return EXIT_SUCCESS;
}


/*
 *  Read in data.
 */

__host__ void load_data(int argc, char **argv, int *K, int **y, float **n){
  int k;
  char line[128];
  FILE *fp;    
    
  if(argc > 1){
    fp = fopen(argv[1], "r");
  } else {
    printf("Please provide input filename\n");
    exit(EXIT_FAILURE);
  }

  if(fp == NULL){
    printf("Cannot read file \n");
    exit(EXIT_FAILURE);
  }

  *K = 0;
  while( fgets (line, sizeof line, fp) != NULL )
    (*K)++; 

  rewind(fp);

  *y = (int*) malloc((*K) * sizeof(int));
  *n = (float*) malloc((*K) * sizeof(float)); 
  
  for(k = 0; k < *K; k++)
    fscanf(fp, "%d %f", *y + k, *n + k);    
 
  fclose(fp);
}


/*
 *  Metropolis algorithm for producing random a values. 
 *  The proposal distribution in normal with a variance that
 *  is adjusted at each step.
 */
 
__host__ float sample_a(float a, float b, int K, float sum_logs){

  static float sigma = 2;
  float U, log_acceptance_ratio, proposal = rnorm() * sigma + a;

  if(proposal <= 0) 
    return a;

  log_acceptance_ratio = (proposal - a) * sum_logs +
                         K * (proposal - a) * log(b) -
                         K * (lgamma(proposal) - lgamma(a));

  U = rand() / float(RAND_MAX);

  if(log(U) < log_acceptance_ratio){
    sigma *= 1.1;
    return proposal;
  } else {
    sigma /= 1.1;
    return a;
  }
}


/*
 *  Sample b from a gamma distribution.
 */

__host__ float sample_b(float a, int K, float flat_sum){

  float hyperA = K * a + 1;
  float hyperB = flat_sum;
  return rgamma(hyperA, hyperB);
}


/* 
 *  Box-Muller Transformation: Generate one standard normal variable.
 *
 *  This algorithm can be more efficiently used by producing two
 *  random normal variables. However, for the CPU, much faster
 *  algorithms are possible (e.g. the Ziggurat Algorithm);
 *
 *  This is actually the algorithm chosen by NVIDIA to calculate
 *  normal random variables on the GPU.
 */
 
__host__ float rnorm(){

  float U1 = rand() / float(RAND_MAX);
  float U2 = rand() / float(RAND_MAX);
  float V1 = sqrt(-2 * log(U1)) * cos(2 * PI * U2);
  /* float V2 = sqrt(-2 * log(U2)) * cos(2 * PI * U1); */
  return V1;
}


/*
 *  See device rgamma function. This is probably not the
 *   fastest way to generate random gamma variables on a CPU.
 */
 
__host__ float rgamma(float a, float b){

  float d = a - 1.0 / 3;
  float Y, U, v;

  while(1){
    Y = rnorm();
    v = pow((1 + Y / sqrt(9 * d)), 3);

    /* Necessary to avoid taking the log of a negative number later. */
    if(v <= 0) 
      continue;
        
    U = rand() / float(RAND_MAX);

    /* Accept the sample under the following condition. 
       Otherwise repeat loop. */
    if(log(U) < 0.5 * pow(Y,2) + d * (1 - v + log(v)))
            return d * v / b;
  }
}


/* 
 *  Generate a single Gamma distributed random variable by the Marsoglia 
 *  algorithm (George Marsaglia, Wai Wan Tsang; 2001).
 *
 *  Zeb chose this algorithm because it has a very high acceptance rate (>96%),
 *  so this while loop will usually only need to run a few times. Many other 
 *  algorithms, while perhaps faster on a CPU, have acceptance rates on the 
 *  order of 50% (very bad in a massively parallel context).
 */

__device__ float rgamma(curandState *state, int id, float a, float b){

  float d = a - 1.0 / 3;
  float Y, U, v;

  while(1){   
    Y = curand_normal(&state[id]);
    v = pow((1 + Y / sqrt(9 * d)), 3);

    /* Necessary to avoid taking the log of a negative number later. */
    if(v <= 0) 
      continue;
        
    U = curand_uniform(&state[id]);

    /* Accept the sample under the following condition. 
       Otherwise repeat loop. */
    if(log(U) < 0.5 * pow(Y,2) + d * (1 - v + log(v)))
      return d * v / b;
  }
}


/*
 *  Sample each theta from the appropriate gamma distribution
 */
 
__global__ void sample_theta(curandState *state, 
                             float *theta, float *log_theta, int *y, float *n, 
                             float a, float b, int K){
                             
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  float hyperA, hyperB;
    
  if(id < K){
    hyperA = a + y[id];
    hyperB = b + n[id];
    theta[id] = rgamma(state, id, hyperA, hyperB);
    log_theta[id] = log(theta[id]);
  }
}


/* 
 *  Initialize GPU random number generators 
 */
 
__global__ void setup_kernel(curandState *state, unsigned int seed, int K){

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  if(id < K)
    curand_init(seed, id, 0, &state[id]);
}