#include <cula.h>
#include <stdlib.h>
#include <stdio.h>

int main(){

  culaStatus s;
  int info;
  char buf[256];

  s = culaInitialize();
  if(s != culaNoError) {
    printf("%s\n", culaGetErrorInfo());
  }

  s = culaSgeqrf(-1, -1, NULL, -1, NULL); /* obviously wrong */ if(s != culaNoError)

  if(s != culaNoError)
  {
    if(s == culaArgumentError)
      printf("Argument %d has an illegal value\n", culaGetErrorInfo());
    else
    {
      info = culaGetErrorInfo(); culaGetErrorInfoString(s, info, buf, sizeof(buf));
      printf("%s", buf);
    }
  }

  culaShutdown();
}