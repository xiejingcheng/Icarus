//https://zhuanlan.zhihu.com/p/324199420
//这个上面关于stockham算法的实现过程讲的比较细致
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../utils/Complex.cuh"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

int getBits(int n) { 
    int bits = 0;
    while (n >>= 1) {
        bits++;
    }
    return bits;
}

__device__ void Bufferfly(Complex *a, Complex *b, Complex factor, Complex *resa, Complex *resb) {
    Complex a1 = (*a) + factor * (*b);
    Complex b1 = (*a) - factor * (*b);
    *resa = a1;
    *resb = b1;
}

//因为下面提到的原因，所以这个函数分为奇偶执行
//这个是偶数的时候的stockham算法
__global__ void FFTEven(Complex nums[], Complex result[], int n, int K) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n/2) return
    //这下面原本打算 判断t的奇偶 但是这样会导致出现分支，使得效率变低
    for(int t = 0; t <= K; t++){
        resx = (tid / (2 << t)) * (2 << (t + 1)) + (tid % (2 << t));
        Bufferfly(&nums[tid], &nums[tid + n/2], Complex::W(), &result[resx], &result[resx + n/2]);

        __syncthreads();

        t++;
        resx = (tid / (2 << t)) * (2 << (t + 1)) + (tid % (2 << t));
        Bufferfly(&result[tid], &result[tid + n/2], Complex::W(), &nums[resx], &nums[resx + n/2]);
        __syncthreads();
    }
    result[tid] = nums[tid];
}

//这个是奇数的时候的stockham算法
__global__ void FFTOdd(Complex nums[], Complex result[], int n, int K) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n/2) return
    for(int t = 0; t <= (K-1); t++){
        resx = (tid / (2 << t)) * (2 << (t + 1)) + (tid % (2 << t));
        Bufferfly(&nums[tid], &nums[tid + n/2], Complex::W(), &result[resx], &result[resx + n/2]);

        __syncthreads();

        t++;
        resx = (tid / (2 << t)) * (2 << (t + 1)) + (tid % (2 << t));
        Bufferfly(&result[tid], &result[tid + n/2], Complex::W(), &nums[resx], &nums[resx + n/2]);
        __syncthreads();
    }
    Bufferfly(&nums[tid], &nums[tid + n/2], Complex::W(), &result[resx], &result[resx + n/2]);
}

int main() {
    srand(time(0));  
    const int TPB = 1024;  
    const int N = 1024 * 32;  
    const int bits = GetBits(N);
    
    Complex *nums = (Complex*)malloc(sizeof(Complex) * N), *dNums, *dResult;
    for (int i = 0; i < N; ++i) {
        nums[i] = Complex::GetRandomReal();
    }
    printf("Length of Sequence: %d\n", N);

    float s = GetTickCount();
    

    cudaMalloc((void**)&dNums, sizeof(Complex) * N);
    cudaMalloc((void**)&dResult, sizeof(Complex) * N);
    cudaMemcpy(dNums, nums, sizeof(Complex) * N, cudaMemcpyHostToDevice);
    
 
    dim3 threadPerBlock = dim3(TPB);
    dim3 blockNum = dim3((N + threadPerBlock.x - 1) / threadPerBlock.x);


    FFT<<<blockNum, threadPerBlock>>>(dNums, dResult, N, bits);

   
    cudaMemcpy(nums, dResult, sizeof(Complex) * N, cudaMemcpyDeviceToHost);
    
   
    float cost = GetTickCount() - s;
    
    printf("Time of Transfromation: %fms", cost);
  
    free(nums);
    cudaFree(dNums);
    cudaFree(dResult);
}