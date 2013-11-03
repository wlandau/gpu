#include <thrust/device_vector.h>
#include <iostream>

int main(){
  // allocate device vector
  thrust::device_vector<int> d_vec(4);
  thrust::device_vector<int>::iterator begin = d_vec.begin();
  thrust::device_vector<int>::iterator end = d_vec.end();
  
  int length = end - begin;
  
  end = d_vec.begin() + 3; // define a sequence of 3 elements
  
  return 0;
}