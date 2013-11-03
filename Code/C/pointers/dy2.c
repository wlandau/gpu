#include <stdio.h>
#include <stdlib.h>

#define M 10
#define N 15

void fill(float *x, int size){
  int i;
  for(i = 0; i < size; ++i){
    x[i] = 10.25 + i*i;
  }
}

int main(){
  int i;
  float *a, *b;
  
  a = (float *) malloc(M * sizeof(float));
  b = (float *) malloc(N * sizeof(float));
  
  fill(a, M);
  fill(b, N);
  
 for(i = 0; i < M; ++i){
    printf("a[%d] = %f\n", i, a[i]);
  }
  printf("\n");
  
  for(i = 0; i < N; ++i){
    printf("b[%d] = %f\n", i, b[i]);
  }
  
  free(a);
  free(b);
}