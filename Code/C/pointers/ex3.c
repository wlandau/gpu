#include <stdio.h>

int main(){
  int a = 0, b = 0;
  int *pa;
  
  pa = &b;
  *pa = a;
  *pa = *pa + 1;
  
  printf("a = %d\n", a);
  printf("&a = %d\n", &a);
  printf("b = %d\n", b);
  printf("&b = %d\n", &b);
  printf("*pa = %d\n", *pa);
  printf("pa = %d\n", pa);
  printf("&pa = %d\n", &pa);
}