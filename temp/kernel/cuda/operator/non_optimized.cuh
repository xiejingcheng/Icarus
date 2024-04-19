#include "../data_error.cuh"
#include <math.h>

extern "C" {
    EXPORT int add_on_device_eq(matrix* data1, matrix* data2, matrix* result);
    EXPORT int sub_on_device_eq(matrix* data1, matrix* data2, matrix* result);
    EXPORT int mul_on_device_eq(matrix* data1, matrix* data2, matrix* result);
    EXPORT int div_on_device_eq(matrix* data1, matrix* data2, matrix* result);  

    __global__ void add_on_device_eq_kernel(float* data1, float* data2, float* result, int len);
    __global__ void sub_on_device_eq_kernel(float* data1, float* data2, float* result, int len);
    __global__ void mul_on_device_eq_kernel(float* data1, float* data2, float* result, int len);
    __global__ void div_on_device_eq_kernel(float* data1, float* data2, float* result, int len);

    EXPORT int sigmoid_on_device(matrix* data1, matrix* result);

    __global__ void sigmoid_on_device_kernel(float* data1, float* result, int len);

    __device__ inline float sigmoid(float x);
};