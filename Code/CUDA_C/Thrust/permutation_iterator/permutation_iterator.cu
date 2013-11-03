#include <thrust/device_vector.h>
#include <thrust/reduce.h> 
#include <thrust/iterator/permutation_iterator.h>
#include <iostream>

int main(){
  thrust::device_vector<int> map(4);
  map[0] = 3;
  map[1] = 1;
  map[2] = 0;
  map[3] = 5;
  
  thrust::device_vector<int> source(6);
  source[0] = 10;
  source[1] = 20;
  source[2] = 30;
  source[3] = 40;
  source[4] = 50;
  source[5] = 60;
  
  typedef thrust::device_vector<int>::iterator indexIter;
  
  thrust::permutation_iterator<indexIter, indexIter> pbegin = 
    thrust::make_permutation_iterator(source.begin(), map.begin());

  thrust::permutation_iterator<indexIter, indexIter> pend = 
    thrust::make_permutation_iterator(source.end(), map.end());
    
  int p0 = pbegin[0]; // source[map[0]] = 40
  int p1 = pbegin[1]; // source[map[1]] = 20
  
  std::cout << "p0 = " << p0 << "\n";
  std::cout << "p1 = " << p1 << "\n";
  
  int sum = thrust::reduce(pbegin, pend);
  
 /* sum = 
  * source [map[0]] + source [map[1]] + source [map[2]] + source [map[3]] = 
  * source [3] + source [1] + source [0] + source [5] = 
  * 40 + 20 + 10 + 60 =
  * 130
  */
  
  std::cout << "sum = " << sum << "\n"; 
  return 0;
}