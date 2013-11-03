all:
	gcc kmeans.c -o kmeans -Wall -ansi -pedantic
	nvcc gpu_kmeans.cu -o gpu_kmeans --compiler-options -ansi --compiler-options -Wall

clean:
	rm kmeans
	rm gpu_kmeans