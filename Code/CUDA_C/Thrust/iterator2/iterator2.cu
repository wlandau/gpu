#include <thrust/device_vector.h>
#include <iostream>

int main(){
  // allocate device vector
  thrust::device_vector<int> d_vec(4);
  thrust::device_vector<int>::iterator begin = d_vec.begin();

  *begin = 13; // same as d_vec[0] = 13; 

  int temp = *begin; // same as temp = d_vec[0];

  begin++; // advance iterator one position 


  *begin = 25; // same as d_vec[1] = 25;
  
  return 0;
}