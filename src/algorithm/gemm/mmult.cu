#include <assert.h>
#include <stdlib.h>

#include "helper.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>



//这个版本比较简单，就是一个线程负责计算结果的一个点。但是每次都是从全局里面取内存，有问题。
//这里需要理解就是二维数组到一纬的转化，需要的是，一维里面是按照行顺序存储的。然而二维数组是 c[y][x] = c[y * n + x]
template <int BLOCK>
__global__ void sgemmV1(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
    //这里不是用 blockDim 因为固定分块大小 这个似乎就是一样的
    int _x = blockIdx.x * BLOCK + threadIdx.x;
    int _y = blockIdx.y * BLOCK + threadIdx.y;
    if (_m < m && _n < n){
        float sum = 0.f;
        for(int i = 0; i < k; i++){
            sum += a[_y * k + i] * b[i * n + _x];
        }
        c[_y * n + _x] = sum;
    }
}


//这个版本使用smem去优化
template <int BLOCK>
__global__ void sgemmV2 (int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    //begin_a定义的是，在
    float *begin_a = a + by * BLOCK * k;
    float *begin_b = b + bx * BLOCK;
    float *end_a = begin_a + k;

    float sum = 0.f;
    //虽然每个block只保存了一部分的矩阵，但是它会让下面的每个线程都计算到这下面的部分 sum 最后存在一起。
    //相当于这个block这个
    //但是还没想好，这个下面是不是全部包含了，会不会还要开第二次shared 内存 应该是可以的，因为他是用 bx by算的小块，应该是可以包括的
    //因为 这个用bx by 计算的小块，所以里面的线程肯定是都用到了小块的，但是会不会需要再开一次就不知道了。
    //应该是不需要的，因为矩阵乘法的时候，不同的元素对都要乘以一次，而且都只乘一次。所以都会覆盖上。
    //这个应该是很对的，因为没有这个的时候，每个block里面都只计算现在的小块里面的东西。
    //同时这个小块也会移动，最后会覆盖所有这个block里面所有的thread需要的矩阵元素

    //但是这里的偏移为什么是这么多我还是不太懂，还要end_a 为什么是这么多
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

void mmult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
    float *d_B, int ldb, float *d_C, int ldc) {

constexpr int BLOCK = 16;
dim3 block(BLOCK, BLOCK);
dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

sgemmV1<BLOCK><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}

int main(){
    printf("sss");
    return 0;
}