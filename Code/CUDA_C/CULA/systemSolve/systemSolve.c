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
 * CULA Example: systemSolve
 *
 * This example shows how to use a system solve for multiple data types.  Each
 * data type has its own example case for clarity.  For each data type, the
 * following steps are done:
 *
 * 1. Allocate a matrix on the host
 * 2. Initialize CULA
 * 3. Initialize the A matrix to the Identity
 * 4. Call gesv on the matrix
 * 5. Verify the results
 * 6. Call culaShutdown
 *
 * After each CULA operation, the status of CULA is checked.  On failure, an
 * error message is printed and the program exits.
 *
 * Note: CULA Premium and double-precision GPU hardware are required to run the
 * double-precision examples
 *
 * Note: this example performs a system solve on an identity matrix against a
 * random vector, the result of which is that same random vector.  This is not
 * true in the general case and is only appropriate for this example.  For a
 * general case check, the product A*X should be checked against B.  Note that
 * because A is modifed by GESV, a copy of A would be needed with which to do
 * the verification.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cula_lapack.h>


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


void culaFloatExample()
{
#ifdef NDEBUG
    int N = 8192;
#else
    int N = 1024;
#endif
    int NRHS = 1;
    int i;

    culaStatus status;
    
    culaFloat* A = NULL;
    culaFloat* B = NULL;
    culaFloat* X = NULL;
    culaInt* IPIV = NULL;

    culaFloat one = 1.0f;
    culaFloat thresh = 1e-6f;
    culaFloat diff;

    printf("-------------------\n");
    printf("       SGESV\n");
    printf("-------------------\n");

    printf("Allocating Matrices\n");
    A = (culaFloat*)malloc(N*N*sizeof(culaFloat));
    B = (culaFloat*)malloc(N*sizeof(culaFloat));
    X = (culaFloat*)malloc(N*sizeof(culaFloat));
    IPIV = (culaInt*)malloc(N*sizeof(culaInt));
    if(!A || !B || !IPIV)
        exit(EXIT_FAILURE);

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    // Set A to the identity matrix
    memset(A, 0, N*N*sizeof(culaFloat));
    for(i = 0; i < N; ++i)
        A[i*N+i] = one;
    
    // Set B to a random matrix (see note at top)
    for(i = 0; i < N; ++i)
        B[i] = (culaFloat)rand();
    memcpy(X, B, N*sizeof(culaFloat));

    memset(IPIV, 0, N*sizeof(culaInt));

    printf("Calling culaSgesv\n");
    status = culaSgesv(N, NRHS, A, N, IPIV, X, N);
    checkStatus(status);

    printf("Verifying Result\n");
    for(i = 0; i < N; ++i)
    {
        diff = X[i] - B[i];
        if(diff < 0.0f)
            diff = -diff;
        if(diff > thresh)
            printf("Result check failed:  i=%d  X[i]=%f  B[i]=%f", i, X[i], B[i]);
    }
    
    printf("Shutting down CULA\n\n");
    culaShutdown();

    free(A);
    free(B);
    free(IPIV);
}


void culaFloatComplexExample()
{
#ifdef NDEBUG
    int N = 4096;
#else
    int N = 512;
#endif
    int NRHS = 1;
    int i;

    culaStatus status;
    
    culaFloatComplex* A = NULL;
    culaFloatComplex* B = NULL;
    culaFloatComplex* X = NULL;
    culaInt* IPIV = NULL;

    culaFloatComplex one = { 1.0f, 0.0f };
    culaFloat thresh = 1e-6f;
    culaFloat diffr;
    culaFloat diffc;
    culaFloat diffabs;

    printf("-------------------\n");
    printf("       CGESV\n");
    printf("-------------------\n");

    printf("Allocating Matrices\n");
    A = (culaFloatComplex*)malloc(N*N*sizeof(culaFloatComplex));
    B = (culaFloatComplex*)malloc(N*sizeof(culaFloatComplex));
    X = (culaFloatComplex*)malloc(N*sizeof(culaFloatComplex));
    IPIV = (culaInt*)malloc(N*sizeof(culaInt));
    if(!A || !B || !IPIV)
        exit(EXIT_FAILURE);

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    // Set A to the identity matrix
    memset(A, 0, N*N*sizeof(culaFloatComplex));
    for(i = 0; i < N; ++i)
        A[i*N+i] = one;
    
    // Set B to a random matrix (see note at top)
    for(i = 0; i < N; ++i)
    {
        B[i].x = (culaFloat)rand();
        B[i].y = (culaFloat)rand();
    }
    memcpy(X, B, N*sizeof(culaFloatComplex));

    memset(IPIV, 0, N*sizeof(culaInt));

    printf("Calling culaCgesv\n");
    status = culaCgesv(N, NRHS, A, N, IPIV, X, N);
    checkStatus(status);

    printf("Verifying Result\n");
    for(i = 0; i < N; ++i)
    {
        diffr = X[i].x - B[i].x;
        diffc = X[i].y - B[i].y;
        diffabs = (culaFloat)sqrt(X[i].x*X[i].x+X[i].y*X[i].y)
                - (culaFloat)sqrt(B[i].x*B[i].x+B[i].y*B[i].y);
        if(diffr < 0.0f)
            diffr = -diffr;
        if(diffc < 0.0f)
            diffc = -diffc;
        if(diffabs < 0.0f)
            diffabs = -diffabs;
        if(diffr > thresh || diffc > thresh || diffabs > thresh)
            printf("Result check failed:  i=%d  X[i]=(%f,%f)  B[i]=(%f,%f)", i, X[i].x, X[i].y, B[i].x, B[i].y);
    }
    
    printf("Shutting down CULA\n\n");
    culaShutdown();

    free(A);
    free(B);
    free(IPIV);
}


// Note: CULA Premium is required for double-precision
#ifdef CULA_PREMIUM
void culaDoubleExample()
{
#ifdef NDEBUG
    int N = 4096;
#else
    int N = 512;
#endif
    int NRHS = 1;
    int i;

    culaStatus status;
    
    culaDouble* A = NULL;
    culaDouble* B = NULL;
    culaDouble* X = NULL;
    culaInt* IPIV = NULL;

    culaDouble one = 1.0;
    culaDouble thresh = 1e-6;
    culaDouble diff;
    
    printf("-------------------\n");
    printf("       DGESV\n");
    printf("-------------------\n");

    printf("Allocating Matrices\n");
    A = (culaDouble*)malloc(N*N*sizeof(culaDouble));
    B = (culaDouble*)malloc(N*sizeof(culaDouble));
    X = (culaDouble*)malloc(N*sizeof(culaDouble));
    IPIV = (culaInt*)malloc(N*sizeof(culaInt));
    if(!A || !B || !IPIV)
        exit(EXIT_FAILURE);

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    // Set A to the identity matrix
    memset(A, 0, N*N*sizeof(culaDouble));
    for(i = 0; i < N; ++i)
        A[i*N+i] = one;
    
    // Set B to a random matrix (see note at top)
    for(i = 0; i < N; ++i)
        B[i] = (culaDouble)rand();
    memcpy(X, B, N*sizeof(culaDouble));

    memset(IPIV, 0, N*sizeof(culaInt));

    printf("Calling culaDgesv\n");
    status = culaDgesv(N, NRHS, A, N, IPIV, X, N);
    if(status == culaInsufficientComputeCapability)
    {
        printf("No Double precision support available, skipping example\n");
        free(A);
        free(B);
        free(IPIV);
        culaShutdown();
        return;
    }
    checkStatus(status);

    printf("Verifying Result\n");
    for(i = 0; i < N; ++i)
    {
        diff = X[i] - B[i];
        if(diff < 0.0)
            diff = -diff;
        if(diff > thresh)
            printf("Result check failed:  i=%d  X[i]=%f  B[i]=%f", i, X[i], B[i]);
    }
    
    printf("Shutting down CULA\n\n");
    culaShutdown();

    free(A);
    free(B);
    free(IPIV);
}


void culaDoubleComplexExample()
{
#ifdef NDEBUG
    int N = 1024;
#else
    int N = 128;
#endif
    int NRHS = 1;
    int i;

    culaStatus status;
    
    culaDoubleComplex* A = NULL;
    culaDoubleComplex* B = NULL;
    culaDoubleComplex* X = NULL;
    culaInt* IPIV = NULL;

    culaDoubleComplex one = { 1.0, 0.0 };
    culaDouble thresh = 1e-6;
    culaDouble diffr;
    culaDouble diffc;
    culaDouble diffabs;

    printf("-------------------\n");
    printf("       ZGESV\n");
    printf("-------------------\n");

    printf("Allocating Matrices\n");
    A = (culaDoubleComplex*)malloc(N*N*sizeof(culaDoubleComplex));
    B = (culaDoubleComplex*)malloc(N*sizeof(culaDoubleComplex));
    X = (culaDoubleComplex*)malloc(N*sizeof(culaDoubleComplex));
    IPIV = (culaInt*)malloc(N*sizeof(culaInt));
    if(!A || !B || !IPIV)
        exit(EXIT_FAILURE);

    printf("Initializing CULA\n");
    status = culaInitialize();
    checkStatus(status);

    // Set A to the identity matrix
    memset(A, 0, N*N*sizeof(culaDoubleComplex));
    for(i = 0; i < N; ++i)
        A[i*N+i] = one;
    
    // Set B to a random matrix (see note at top)
    for(i = 0; i < N; ++i)
    {
        B[i].x = (culaDouble)rand();
        B[i].y = (culaDouble)rand();
    }
    memcpy(X, B, N*sizeof(culaDoubleComplex));

    memset(IPIV, 0, N*sizeof(culaInt));

    printf("Calling culaZgesv\n");
    status = culaZgesv(N, NRHS, A, N, IPIV, X, N);
    if(status == culaInsufficientComputeCapability)
    {
        printf("No Double precision support available, skipping example\n");
        free(A);
        free(B);
        free(IPIV);
        culaShutdown();
        return;
    }
    checkStatus(status);

    printf("Verifying Result\n");
    for(i = 0; i < N; ++i)
    {
        diffr = X[i].x - B[i].x;
        diffc = X[i].y - B[i].y;
        diffabs = (culaDouble)sqrt(X[i].x*X[i].x+X[i].y*X[i].y)
                - (culaDouble)sqrt(B[i].x*B[i].x+B[i].y*B[i].y);
        if(diffr < 0.0)
            diffr = -diffr;
        if(diffc < 0.0)
            diffc = -diffc;
        if(diffabs < 0.0)
            diffabs = -diffabs;
        if(diffr > thresh || diffc > thresh || diffabs > thresh)
            printf("Result check failed:  i=%d  X[i]=(%f,%f)  B[i]=(%f,%f)", i, X[i].x, X[i].y, B[i].x, B[i].y);
    }
    
    printf("Shutting down CULA\n\n");
    culaShutdown();

    free(A);
    free(B);
    free(IPIV);
}
#endif


int main(int argc, char** argv)
{
    culaFloatExample();
    culaFloatComplexExample();
    
    // Note: CULA Premium is required for double-precision
#ifdef CULA_PREMIUM
    culaDoubleExample();
    culaDoubleComplexExample();
#endif

    return EXIT_SUCCESS;
}

