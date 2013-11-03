kmeans.c: naive 2-dimensional implementation of Lloyd's K-means algorithm
gpu_kmeans.cu: parallelized CUDA C implementation of the same.

Required files:

  x.txt: x coordinates of the data points arranged in a column.
  x.txt: x coordinates of the data points arranged in a column (same length as in x.txt).

  mu_x.txt: initial cluster centers in the x direction arranged in a column.
  mu_y.txt: initial cluster centers in the y direction arranged in a column (same length as mu_y.txt).


Compilation of kmeans.c:

  > gcc kmeans.c -o kmeans

Compilation of gpu_kmeans.cu:

  > nvcc gpu_kmeans.cu -o gpu_kmeans

Compile both with:

  > make


Run:

  > ./kmeans

  or

  > ./gpu_kmeans


Output:

  mu_x_out.txt: final cluster centers in the x direction
  mu_y_out.txt: final cluster centers in the y direction
  group_out.txt: final cluster index of each point  


Plots (kmeans-before.pdf and kmeans-after.pdf):

  > R CMD BATCH plots.r 

