#include <stdio.h>

int main(){
  int i = 0, *a;
  *a = i;
  printf("*a = %d\n", *a);
  
  *(a + 10000) = 1;
}