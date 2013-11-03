#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <iostream>

int main(){
  // create iterators
  thrust::counting_iterator<int> first(10);
  thrust::counting_iterator<int> last = first + 3;
  
  std::cout << first[0] << "\n"; // returns 10
  std::cout << first[1] << "\n"; // returns 10
  std::cout << first[100] << "\n"; // returns 100
  
  // sum of [first, last)
  std::cout << thrust::reduce(first, last) << "\n"; // returns 33;
}