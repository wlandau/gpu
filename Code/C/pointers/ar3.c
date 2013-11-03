#include <stdio.h>

int main(){
  int i;
  int pa[4]; // declares an array with 4 elements
  
  *pa = 9;        // same as pa[0] = 9
  *(pa + 1) = 17; // same as pa[1] = 17 
  *(pa + 2) = 25; // same as pa[2] = 25
  *(pa + 3) = 7;   // same as pa[3] = 7
  
  printf("%d\n", *pa); // prints the value 9
  printf("%d\n", *(pa + 1)); // prints the value 17
  printf("%d\n", *(pa + 2)); // prints the value 25
  printf("%d\n", *(pa + 3)); // prints the value 7
  printf("pa = %d\n", pa);
}
