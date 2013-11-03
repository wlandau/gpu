#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <iostream>

int main(){
  int sum;
  thrust::device_vector <int> D(10, 1);
  
  sum = thrust::reduce(D.begin(), D.end(), (int) 0, thrust::plus<int>());
  sum = thrust::reduce(D.begin(), D.end(), (int) 0);
  sum = thrust::reduce(D.begin(), D.end());
  
  std::cout << "sum = " << sum << "\n";
  
  // put three 1s in a device vector
  thrust::device_vector <int> vec(5,0);
  vec[1] = 1; vec[3] = 1; vec[4] = 1;
  
  int result = thrust::count(vec.begin(), vec.end() , 1);
  // result is three
  
  std::cout << "result = " << result << "\n";
  
  return 0;
}