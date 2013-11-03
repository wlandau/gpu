#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <iostream>

#include <thrust/iterator/constant_iterator.h>

int main(){
  // create iterators
  thrust::constant_iterator<int> first(10);
  thrust::constant_iterator<int> last = first + 3;
  
  std::cout << first[0] << "\n"; // returns 10
  std::cout << first[1] << "\n"; // returns 10
  std::cout << first[100] << "\n"; // returns 100
  
  // sum of [first, last)
  std::cout << thrust::reduce(first, last) << "\n"; // returns 30;
}