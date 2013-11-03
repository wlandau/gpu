#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>

int main(){
  thrust::device_vector<int> int_v(3);
  int_v[0] = 0; int_v[1] = 1; int_v[2] = 2;

  thrust::device_vector<float> float_v(3);
  float_v[0] = 0.0; float_v[1] = 1.0; float_v[2] = 2.0;

  thrust::device_vector<char> char_v(3);
  char_v[0] = 'a'; char_v[1] = 'b'; char_v[2] = 'c';

  // typedef these iterators for shorthand
  typedef thrust::device_vector<int>::iterator   IntIterator;
  typedef thrust::device_vector<float>::iterator FloatIterator;
  typedef thrust::device_vector<char>::iterator  CharIterator;

  // typedef a tuple of these iterators
  typedef thrust::tuple<IntIterator, FloatIterator, CharIterator> IteratorTuple;

  // typedef the zip_iterator of this tuple
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

  // finally, create the zip_iterator
  ZipIterator iter(thrust::make_tuple(int_v.begin(), float_v.begin(), char_v.begin()));

  *iter;   // returns (0, 0.0, 'a')
  iter[0]; // returns (0, 0.0, 'a')
  iter[1]; // returns (1, 1.0, 'b')
  iter[2]; // returns (2, 2.0, 'c')

  thrust::get<0>(iter[2]); // returns 2
  thrust::get<1>(iter[0]); // returns 0.0
  thrust::get<2>(iter[1]); // returns 'b'

  // iter[3] is an out-of-bounds error
}