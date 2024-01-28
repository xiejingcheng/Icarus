//https://blog.csdn.net/tiger1334/article/details/44703317

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include<assert.h>
#define SHARED_SIZE_LIMIT 1024U
#define    UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )
typedef unsigned int uint;
 
__device__ inline void Comparator(
    uint &keyA,
    uint &valA,
    uint &keyB,
    uint &valB,
    uint dir
)
{
    uint t;
 
    if ((keyA > keyB) == dir)
    {
        t = keyA;
        keyA = keyB;
        keyB = t;
        t = valA;
        valA = valB;
        valB = t;
    }
}
__global__ void bitonicSortShared(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint dir
)
{
    //Shared memory storage for one or more short vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];
 
    //Offset to the beginning of subbatch and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
 
    for (uint size = 2; size < arrayLength; size <<= 1)
    {
        //Bitonic merge
        uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);
 
        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                ddd
            );
        }
    }
 
    //ddd == dir for the last bitonic merge step
    {
        for (uint stride = arrayLength / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                dir
            );
        }
    }
 
    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}
 
 
 
 
// Bitonic sort kernel for large arrays (not fitting into shared memory)
 
//Bottom-level bitonic sort
//Almost the same as bitonicSortShared with the exception of
//even / odd subarrays being sorted in opposite directions
//Bitonic merge accepts both
//Ascending | descending or descending | ascending sorted pairs
__global__ void bitonicSortShared1(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal
)
{
    //Shared memory storage for current subarray
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];
 
    //Offset to the beginning of subarray and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
 
    for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1)
    {
        //Bitonic merge
        uint ddd = (threadIdx.x & (size / 2)) != 0;
 
        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                ddd
            );
        }
    }
 
    //Odd / even arrays of SHARED_SIZE_LIMIT elements
    //sorted in opposite directions
    uint ddd = blockIdx.x & 1;
    {
        for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                ddd
            );
        }
    }
 
 
    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}
 
//Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void bitonicMergeGlobal(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint size,
    uint stride,
    uint dir
)
{
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);
 
    //Bitonic merge
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));
 
    uint keyA = d_SrcKey[pos +      0];
    uint valA = d_SrcVal[pos +      0];
    uint keyB = d_SrcKey[pos + stride];
    uint valB = d_SrcVal[pos + stride];
 
    Comparator(
        keyA, valA,
        keyB, valB,
        ddd
    );
 
    d_DstKey[pos +      0] = keyA;
    d_DstVal[pos +      0] = valA;
    d_DstKey[pos + stride] = keyB;
    d_DstVal[pos + stride] = valB;
}
 
//Combined bitonic merge steps for
//size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeShared(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint size,
    uint dir
)
{
    //Shared memory storage for current subarray
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];
 
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
 
    //Bitonic merge
    uint comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
 
    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(
            s_key[pos +      0], s_val[pos +      0],
            s_key[pos + stride], s_val[pos + stride],
            ddd
        );
    }
 
    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}
 
 
 
 
// Interface function
 
//Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L)
{
    if (!L)
    {
        *log2L = 0;
        return 0;
    }
    else
    {
        for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);
 
        return L;
    }
}
 
extern "C" uint bitonicSort(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint batchSize,
    uint arrayLength,
    uint dir
)
{
    //Nothing to sort
    if (arrayLength < 2)
        return 0;
 
    //Only power-of-two array lengths are supported by this implementation
    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);
 
    dir = (dir != 0);
 
    uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;
 
    if (arrayLength <= SHARED_SIZE_LIMIT)
    {
        assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
        bitonicSortShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
    }
    else
    {
        bitonicSortShared1<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal);
 
        for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1)
                if (stride >= SHARED_SIZE_LIMIT)
                {
                    bitonicMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, stride, dir);
                }
                else
                {
                    bitonicMergeShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, dir);
                    break;
                }
    }
 
    return threadCount;
}
int main(){
const uint             N = 1048576;
    const uint           DIR = 0;
    const uint     numValues = 65536;
 
uint *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;
    uint *d_InputKey, *d_InputVal,    *d_OutputKey,    *d_OutputVal;
h_InputKey     = (uint *)malloc(N * sizeof(uint));
    h_InputVal     = (uint *)malloc(N * sizeof(uint));
    h_OutputKeyGPU = (uint *)malloc(N * sizeof(uint));
    h_OutputValGPU = (uint *)malloc(N * sizeof(uint));
 
	   for (uint i = 0; i < N; i++)
    {
        h_InputKey[i] = rand() % numValues;
        h_InputVal[i] = i;
    }
	 cudaMalloc((void **)&d_InputKey,  N * sizeof(uint));
 cudaMalloc((void **)&d_InputVal,  N * sizeof(uint));
 cudaMalloc((void **)&d_OutputKey, N * sizeof(uint));
 cudaMalloc((void **)&d_OutputVal, N * sizeof(uint));
 cudaMemcpy(d_InputKey, h_InputKey, N * sizeof(uint), cudaMemcpyHostToDevice);
 cudaMemcpy(d_InputVal, h_InputVal, N * sizeof(uint), cudaMemcpyHostToDevice);
	   
	   int flag = 1;
	   for(uint arrayLength = 64;arrayLength<=N;arrayLength*=2){
		   cudaDeviceSynchronize();
		   bitonicSort(d_OutputKey,d_OutputVal,d_InputKey,d_InputVal,
			   N/arrayLength,arrayLength,DIR);
		   cudaDeviceSynchronize();
	   }
	    cudaMemcpy(h_OutputKeyGPU, d_OutputKey, N * sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_OutputValGPU, d_OutputVal, N * sizeof(uint), cudaMemcpyDeviceToHost);
		for(int i=0;i<N;i++){
		printf("%d %d\n",h_OutputKeyGPU[i],h_OutputValGPU[i]);
		}
		 cudaFree(d_OutputVal);
         cudaFree(d_OutputKey);
         cudaFree(d_InputVal);
         cudaFree(d_InputKey);
		 free(h_OutputValGPU);
		 free(h_OutputKeyGPU);
		 free(h_InputVal);
		 free(h_InputKey);
		 system("pause");
		 return 0;
}