#include "../data_error.cuh"

extern "C" {
    EXPORT int gemm_on_device_2d(tensor* data1, tensor* data2, tensor* result);
};

template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, float *b, float *c);