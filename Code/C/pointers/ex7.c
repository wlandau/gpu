#include <stdio.h>

void fun(int *a){
  *a = *a + 1; 
}

int main(){
  int a = 0, *pa;
  
  pa = &a; 
  fun(pa);
  
  printf("a = %d\n", a);
  printf("*pa = %d\n", *pa);
}