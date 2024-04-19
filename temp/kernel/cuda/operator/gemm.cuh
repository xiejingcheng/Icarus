#include "../data_error.cuh"

extern "C" {
    EXPORT int gemm_on_device_2d(matrix* data1, matrix* data2, matrix* result);
};

template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, float *b, float *c);