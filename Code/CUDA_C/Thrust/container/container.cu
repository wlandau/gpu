#include <thrust/device_vector.h>
#include <thrust/copy.h> 
#include <list>
#include <vector>

int main(void){
  // create an STL list with 4 values
  std::list<int> stl_list;
  stl_list.push_back(10);
  stl_list.push_back(20); 
  stl_list.push_back(30);
  stl_list.push_back(40);

  // initialize a device_vector with the list
  thrust::device_vector <int> D(stl_list.begin(), stl_list.end());

  // copy a device_vector into an STL vector
  std::vector<int> stl_vector(D.size()); 
  thrust::copy(D.begin(), D.end(), stl_vector.begin());

  return 0;
}