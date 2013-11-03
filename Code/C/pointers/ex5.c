#include <stdio.h>

void fun(int *a){
  *a = *a + 1; 
}

int main(){
  int a = 0;
  
  fun(&a);
  
  printf("a = %d\n", a);
}