COMPUTE=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
INC_DIR=/usr/local/cula/include
LIB_DIR=${CULA_LIB_PATH_64}
LIBS=-lcula_lapack -lcula_lapack_link -lcublas -liomp5

all:
	nvcc ${COMPUTE} -I${INC_DIR} -L${LIB_DIR} ${LIBS} ols.cu -o ols

clean:
	rm ols