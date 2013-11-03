#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <iostream>

int main(){
  int data[6] = {1, 0, 2, 2, 1, 3};
  thrust::inclusive_scan(data, data + 6, data); 

 /* data[0] = data[0]
  * data[1] = data[0] + data[1]
  * data[2] = data[0] + data[1] + data[2]
  * ...
  * data[5] = data[0] + data[1] + ... + data[5]
  */
  
  // data is now {1, 1, 3, 5, 6, 9}
}