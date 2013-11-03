Created by Zebulun Arendsee.
March 26, 2013

Modified by Will Landau.
June 30, 2013
will-landau.com
landau@iastate.edu

This program implements a MCMC algorithm for the following hierarchical
model:

y_k     ~ Poisson(n_k * theta_k)     k = 1, ..., K
theta_k ~ Gamma(a, b)
a       ~ Unif(0, a0)
b       ~ Unif(0, b0) 

We let a0 and b0 be arbitrarily large.

Arguments:
    1) input filename
        With two space delimited columns holding integer values for
        y and float values for n.
    2) number of trials (1000 by default)

Output: A comma delimited file containing a column for a, b, and each
theta. All output is written to standard out.

Example dataset:

$ head -3 data.txt
4 0.91643
23 3.23709
7 0.40103

Example of compilation and execution:

$ nvcc gibbs_metropolis.cu -o gibbs
$ ./gibbs mydata.txt 2500 > output.csv
$

This code borrows from the nVidia developer zone documentation, 
specifically http://docs.nvidia.com/cuda/curand/index.html#topic_1_2_1