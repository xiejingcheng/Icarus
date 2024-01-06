#include "../../utils/Complex.cuh"

int getBits(int n){
    int bits = 0;
    while(n >>= 1){
        bits++;
    }
    return bits;
}

__device__ int binaryReverse(int i, int bits) {
    int r = 0;
    do {
        r += i % 2 << --bits;
    } while (i /= 2);
    return r;
}

__device__ void bufferfly(Complex *a, Complex *b, Complex factor) {
    Complex a1 = (*a) + factor * (*b);
    Complex b1 = (*a) - factor * (*b);
    *a = a1;
    *b = b1;
}

__global__ void Reduce(int nums[], int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) {
            nums[tid] += nums[tid + i / 2];
        }
        __syncthreads();
    }
}

__global__ void FFTCooleyTurkeyV1(Complex nums[], Complex result[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) {
            int k = i;
            if (n - tid < k) k = n - tid;
            for (int j = 0; j < k / 2; ++j) {
                Bufferfly(&nums[BinaryReverse(tid + j, bits)], &nums[BinaryReverse(tid + j + k / 2, bits)], Complex::W(k, j));
            }
        }
        __syncthreads();
    }
    result[tid] = nums[BinaryReverse(tid, bits)];
}

// __global__ void reverseNums(Complex nums[], Complex revnums[], int n, int bits) {
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;
//     if (tid <= n){
//         revnums[tid] = nums[BinaryReverse(tid, bits)];
//     }
//     __syncthreads();
// }

//我的想法是原地给他排好序号，但是我不知道怎么找对应的编号
//按照我的想法应该可以降低到O(log n)
__global__ void FFTCooleyTurkeyV2(Complex nums[], Complex result[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid <= n){
        result[tid] = nums[BinaryReverse(tid, bits)];
    }
    __syncthreads();

    // for (int i = 1; i <= 2 * n; i *= 2) {
    //     if()
    // }

}

