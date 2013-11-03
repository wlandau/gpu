#include <cula.h>
#include <stdlib.h>
#include <stdio.h>

int main(){

  culaStatus s;
  int info;
  char buf[256];

  float* A = (float *) malloc(20 * 20 * sizeof(float));
  memset(A, 0, 20*20*sizeof(float)); /* singular matrix, illegal for LU (getrf) */ 
  int ipiv[20];

  s = culaInitialize();
  if(s != culaNoError) {
    printf("%s\n", culaGetErrorInfo());
  }

  s = culaSgetrf(20, 20, A, 20, ipiv);

  if( s != culaNoError )
  {
    if( s == culaDataError )
      printf("Data error with code %d, please see LAPACK documentation\n", culaGetErrorInfo());
    else
    {
      info = culaGetErrorInfo(); culaGetErrorInfoString(s, info, buf, sizeof(buf));
      printf("%s", buf);
    }
  }    

  culaShutdown();
}