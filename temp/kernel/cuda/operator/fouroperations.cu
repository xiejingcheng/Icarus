#include "fouroperations.cuh"
#include <cuda_runtime.h>

EXPORT int add_on_device_eq(tensor* data1, tensor* data2, tensor* result){
    int len = 1;
    for (int i = 0; i < data1->dim; i++){
        len *= data1->size[i];
    }

    dim3 block(256);
    dim3 grid((len + 256 - 1) / 256);
    add_on_device_eq_kernel<<<grid, block>>>(data1->data_device, data2->data_device, result->data_device, len);
    return 0;
}

EXPORT int sub_on_device_eq(tensor* data1, tensor* data2, tensor* result){
    int len = 1;
    for (int i = 0; i < data1->dim; i++){
        len *= data1->size[i];
    }

    dim3 block(256);
    dim3 grid((len + 256 - 1) / 256);
    sub_on_device_eq_kernel<<<grid, block>>>(data1->data_device, data2->data_device, result->data_device, len);
    return 0;
}

EXPORT int mul_on_device_eq(tensor* data1, tensor* data2, tensor* result){
    int len = 1;
    for (int i = 0; i < data1->dim; i++){
        len *= data1->size[i];
    }

    dim3 block(256);
    dim3 grid((len + 256 - 1) / 256);
    mul_on_device_eq_kernel<<<grid, block>>>(data1->data_device, data2->data_device, result->data_device, len);
    return 0;
}

EXPORT int div_on_device_eq(tensor* data1, tensor* data2, tensor* result){
    int len = 1;
    for (int i = 0; i < data1->dim; i++){
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

