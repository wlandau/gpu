#include <stdio.h>

int main(){
  int a = 0;
  int *pa;
  
  pa = &a;
  *pa = *pa + 1;
  
  printf("a = %d\n", a);
  printf("&a = %d\n", &a);
  printf("*pa = %d\n", *pa);
  printf("pa = %d\n", pa);
  printf("&pa = %d\n", &pa);
}