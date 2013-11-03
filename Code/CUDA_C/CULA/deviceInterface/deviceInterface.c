/*
 * Copyright (C) 2009-2012 EM Photonics, Inc.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to EM Photonics ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code may
 * not redistribute this code without the express written consent of EM
 * Photonics, Inc.
 *
 * EM PHOTONICS MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  EM PHOTONICS DISCLAIMS ALL WARRANTIES WITH REGARD TO
 * THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL EM
 * PHOTONICS BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
 * DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as that
 * term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of "commercial
 * computer  software"  and "commercial computer software documentation" as
 * such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the
 * U.S. Government only as a commercial end item.  Consistent with 48
 * C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the source code with only those rights set
 * forth herein. 
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code, the
 * above Disclaimer and U.S. Government End Users Notice.
 *
 */

/*
 * CULA Example: deviceInterface
 *
 * This example is meant to show a typical program flow when using CULA's device
 * interface.
 *
 * 1. Allocate matrices
 * 2. Initialize CULA
 * 3. Initialize the A matrix to zero
 * 4. Copy the A matrix to the device Ad matrix
 * 4. Call culaDeviceSgeqrf (QR factorization) on the matrix Ad
 * 5. Copies the results (Ad,TAUd) back to the host (A,TAU)
 * 6. Call culaShutdown
 *
 * After each CULA operation, the status of CULA is checked.  On failure, an
 * error message is printed and the program exits.
 *
 * Note: this example performs the QR factorization on a matrix of zeros, the
 * result of which is a matrix of zeros, due to the omission of ones across the
 * diagonal of the upper-diagonal unitary Q.
 *
 * Note: this example requires the CUDA toolkit to compile.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cula_lapack_device.h>

#include <cuda_runtime.h>
#ifdef _MSC_VER
#   pragma comment(lib, "cudart.lib")
#endif


void checkStatus(culaStatus status)
{
    char buf[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    exit(EXIT_FAILURE);
}


void checkCudaError(cudaError_t err)
{
    if(!err)
        return;

    printf("%s\n", cudaGetErrorString(err));

    culaShutdown();
    exit(EXIT_FAILURE);
}


int main(int argc, char** argv)
{
#ifdef NDEBUG
    int M = 8192;
#else
    int M = 1024;
#endif
    int N = M;

    cudaError_t err;
    culaStatus status;

    // point to host memory
    float* A = NULL;
    float* TAU = NULL;

    // point to device memory
    float* Ad = NULL;
    float* TAUd = NULL;

    printf("Allocating Matrices\n");
    A = (float*)malloc(M*N*sizeof(float));
    TAU = (float*)malloc(N*sizeof(float));
    if(!A || !TAU)
        exit(EXIT_FAILURE);

    err = cudaMalloc((void**)&Ad, M*N*sizeof(float));
    checkCudaError(err);

    err = cudaMalloc((void**)&TAUd, N*sizeof(float));
    checkCudaError(err);

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    memset(A, 0, M*N*sizeof(float));
    err = cudaMemcpy(Ad, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err);

    printf("Calling culaDeviceSgeqrf\n");
    status = culaDeviceSgeqrf(M, N, Ad, M, TAUd);
    checkStatus(status);

    err = cudaMemcpy(A, Ad, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err);
    err = cudaMemcpy(TAU, TAUd, N*sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err);

    printf("Shutting down CULA\n");
    culaShutdown();

    cudaFree(Ad);
    cudaFree(TAUd);
    free(A);
    free(TAU);

    return EXIT_SUCCESS;
}

