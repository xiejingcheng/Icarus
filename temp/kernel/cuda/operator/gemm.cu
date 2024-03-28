#include "gemm.cuh"
#include <cuda_runtime.h>

EXPORT int gemm_on_device_2d(tensor* data1, tensor* data2, tensor* result){
    constexpr int BLOCK = 16;
    int m = data1->size[0];
    int n = data2->size[1];
    int k;
    if (data1->size[1] != data2->size[0]){
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }else {
        k = data1->size[1];
    }

    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    sgemm<BLOCK><<<grid, block>>>(m, n, k, data1->data_device, data2->data_device, result->data_device);

    return 0;
}


//行优先存储 x代表列索引 y代表行索引
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, float *b, float *c) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    float *begin_a = a + by * BLOCK * k;
    float *begin_b = b + bx * BLOCK;
    float *end_a = begin_a + k;

    float sum = 0.f;

    for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
        a_ptr += BLOCK, b_ptr += BLOCK * n) {
        __shared__ float ashare[BLOCK][BLOCK];
        __shared__ float bshare[BLOCK][BLOCK];
    
        ashare[ty][tx] = a_ptr[ty * k + tx];
        bshare[ty][tx] = b_ptr[ty * n + tx];
        __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BLOCK; ++kk) {
       sum += ashare[ty][kk] * bshare[kk][tx];
    }
     __syncthreads();
    }
 
   c[(BLOCK * by + ty) * n + BLOCK * bx + tx] = sum;
 
}

