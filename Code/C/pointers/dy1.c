#include <stdio.h>
#include <stdlib.h>

void fill(int *a){
  int i;
  for(i = 0; i < 10; ++i){
    a[i] = 10 + i*i;
  }
}

int main(){
  int i, *a;
  
  a = (int *) malloc(10 * sizeof(int));
  fill(a);
  
 for(i = 0; i < 10; ++i){
    printf("a[%d] = %d\n", i, a[i]);
  }
  
  free(a);
}