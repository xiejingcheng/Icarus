#include "non_optimized.cuh"
#include <cuda_runtime.h>

EXPORT int add_on_device_eq(matrix* data1, matrix* data2, matrix* result){
    int len = 1;
    for (int i = 0; i < data1->dims; i++){
        len *= data1->size[i];
    }

    dim3 block(256);
    dim3 grid((len + 256 - 1) / 256);
    add_on_device_eq_kernel<<<grid, block>>>(data1->data_device, data2->data_device, result->data_device, len);
    return 0;
}

EXPORT int sub_on_device_eq(matrix* data1, matrix* data2, matrix* result){
    int len = 1;
    for (int i = 0; i < data1->dims; i++){
        len *= data1->size[i];
    }

    dim3 block(256);
    dim3 grid((len + 256 - 1) / 256);
    sub_on_device_eq_kernel<<<grid, block>>>(data1->data_device, data2->data_device, result->data_device, len);
    return 0;
}

EXPORT int mul_on_device_eq(matrix* data1, matrix* data2, matrix* result){
    int len = 1;
    for (int i = 0; i < data1->dims; i++){
        len *= data1->size[i];
    }

    dim3 block(256);
    dim3 grid((len + 256 - 1) / 256);
    mul_on_device_eq_kernel<<<grid, block>>>(data1->data_device, data2->data_device, result->data_device, len);
    return 0;
}

EXPORT int div_on_device_eq(matrix* data1, matrix* data2, matrix* result){
    int len = 1;
    for (int i = 0; i < data1->dims; i++){
        len *= data1->size[i];
    }

    dim3 block(256);
    dim3 grid((len + 256 - 1) / 256);
    div_on_device_eq_kernel<<<grid, block>>>(data1->data_device, data2->data_device, result->data_device, len);
    return 0;
}

__global__ void add_on_device_eq_kernel(float* data1, float* data2, float* result, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len){
        result[idx] = data1[idx] + data2[idx];
    }
}

__global__ void sub_on_device_eq_kernel(float* data1, float* data2, float* result, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len){
        result[idx] = data1[idx] - data2[idx];
    }
}

__global__ void mul_on_device_eq_kernel(float* data1, float* data2, float* result, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len){
        result[idx] = data1[idx] * data2[idx];
    }
}

__global__ void div_on_device_eq_kernel(float* data1, float* data2, float* result, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len){
        result[idx] = data1[idx] / data2[idx];
    }
}

EXPORT int sigmoid_on_device(matrix* data1, matrix* result){
    int len = 1;
    for (int i = 0; i < data1->dims; i++){
        len *= data1->size[i];
    }

    dim3 block(256);
    dim3 grid((len + 256 - 1) / 256);
    sigmoid_on_device_kernel<<<grid, block>>>(data1->data_device, result->data_device, len);
    return 0;
}

__global__ void sigmoid_on_device_kernel(float* data1, float* result, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len){
        result[idx] = sigmoid(data1[idx]);
    }
}

__device__ inline float sigmoid(float x){
    return 1 / (1 + exp(-x));
}

