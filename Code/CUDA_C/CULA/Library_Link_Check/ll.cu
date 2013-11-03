#include <cula.h>
#include <stdlib.h>
#include <stdio.h>

int MeetsMinimumCulaRequirements() {

  int cudaMinimumVersion = culaGetCudaMinimumVersion();
  int cudaRuntimeVersion = culaGetCudaRuntimeVersion();
  int cudaDriverVersion = culaGetCudaDriverVersion();
  int cublasMinimumVersion = culaGetCublasMinimumVersion(); 
  int cublasRuntimeVersion = culaGetCublasRuntimeVersion();

  if(cudaRuntimeVersion < cudaMinimumVersion) {
    printf("CUDA runtime version is insufficient;\nversion %d or greater is required\n", cudaMinimumVersion);
    return 0; }

  if(cudaDriverVersion < cudaMinimumVersion) {
    printf("CUDA driver version is insufficient;\nversion %d or greater is required\n", cudaMinimumVersion);
    return 0; 
  }

  if(cublasRuntimeVersion < cublasMinimumVersion) {
    printf("CUBLAS runtime version is insufficient;\nversion %d or greater is required\n", cublasMinimumVersion);
    return 0; 
  }

  return 1; 
}

int main(){
  MeetsMinimumCulaRequirements();
}