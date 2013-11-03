#include <cula.h>
#include <stdlib.h>
#include <stdio.h>

int main(){
  
  culaStatus s;
  s = culaInitialize();
  
  if(s != culaNoError)
    printf("%s\n", culaGetErrorInfo());
    
  /* ... Your code ... */
  
  culaShutdown();
}