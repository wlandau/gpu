#include <thrust/functional.h>
#include <thrust/sort.h>
#include <iostream>

int main(){
  const int N = 6;
  int A[N] = {1, 4, 2, 8, 5, 7}; 
  thrust::sort(A, A+N);
  // A is now {1, 2, 4, 5, 7, 8}

  int keys[N] = { 1, 4, 2, 8, 5, 7}; 
  char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
  thrust::sort_by_key(keys , keys + N, values);
  // keys is now { 1, 2, 4, 5, 7, 8}
  // values is now {'a', 'c', 'b', 'e', 'f', 'd'}

  int A2[N] = {1, 4, 2, 8, 5, 7};
  thrust::stable_sort(A2, A2 + N, thrust::greater<int>());
  // A2 is now {8, 7, 5, 4, 2, 1}
  
  return 0;
}