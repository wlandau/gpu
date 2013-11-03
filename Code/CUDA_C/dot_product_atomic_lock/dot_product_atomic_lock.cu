#include "../common/book.h" 
#include "../common/lock.h"

#define imin(a,b) (a<b?a:b)

// NOTE: COMPILE LIKE THIS:
//nvcc dot_product_atomic.cu -arch sm_20 -o dot_product_atomic

const int N = 32 * 1024 * 1024; 
const int threadsPerBlock = 256; 
const int blocksPerGrid =
  imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );
  
__global__ void dot( Lock lock, float *a, 
                     float *b, float *c ) {
                     
  __shared__ float cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x; 
  int cacheIndex = threadIdx.x;
  
  float temp = 0; 
  while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
  }
    
  // set the cache values
  cache[cacheIndex] = temp;

  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2 
  // because of the following code
  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2; 
  }
  
  // Here's where locks come in:
  if (cacheIndex == 0) { lock.lock();
        *c += cache[0];
        lock.unlock();
  }
}

int main( void ) {
  float *a, *b, c = 0;
  float *dev_a, *dev_b, *dev_c;

  // allocate memory on the CPU side
  a = (float*)malloc( N*sizeof(float) ); 
  b = (float*)malloc( N*sizeof(float) );
  
  // allocate the memory on the GPU
  HANDLE_ERROR( cudaMalloc( (void**)&dev_a, 
                            N*sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_b, 
                            N*sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_c, 
                            sizeof(float) ) );

  // fill in the host memory with data
  for (int i=0; i<N; i++) { 
    a[i] = i;
    b[i] = i*2; 
  }
  
  // copy the arrays 'a' and 'b' to the GPU
  HANDLE_ERROR( cudaMemcpy( dev_a, a, N*sizeof(float), 
                            cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( dev_b, b, N*sizeof(float), 
                            cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( dev_c, &c, sizeof(float), 
                            cudaMemcpyHostToDevice ) );

  Lock lock;
  dot<<<blocksPerGrid,threadsPerBlock>>>( lock, dev_a,
                                          dev_b, dev_c );
  
  // copy c back from the GPU to the CPU
  HANDLE_ERROR( cudaMemcpy( &c, dev_c, 
                            sizeof(float),
                            cudaMemcpyDeviceToHost ) );
                                                      
  #define sum_squares(x) (x*(x+1)*(2*x+1)/6) 
  printf( "Does GPU value %.6g = %.6g?\n", c,
           2 * sum_squares( (float)(N - 1) ) );
           
  // free memory on the GPU side
  cudaFree( dev_a );
  cudaFree( dev_b );
  cudaFree( dev_c );

  // free memory on the CPU side
  free( a );
  free( b ); 
}
                            
                            