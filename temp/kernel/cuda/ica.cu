#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils/device.cuh"
#include "utils/random.cuh"
#include "operator/gemm.cuh"
#include "data_error.cuh"
#include "oi/print.cuh"
#include "operator/fouroperations.cuh"

extern "C" {

    //检查最后一个错误，并且打印错误信息
    inline bool checkCUDAError() {
        cudaError_t err = cudaGetLastError();
    
        if (cudaSuccess != err)
            printf("%s\n", cudaGetErrorString( err));
        return cudaSuccess != err;
    }

    EXPORT const char* get_last_cuda_error() {
        cudaError_t err = cudaGetLastError();
    
        return cudaGetErrorString( err);
    }

    EXPORT const char* get_last_clib_error() {
        return strerror(errno);
    }

    EXPORT int cuda_set_device(int deviceId) {
        cudaSetDevice(deviceId);
        
        if (checkCUDAError())
            return CUDA_ERROR;
        else
            return 0;
    }

    EXPORT void set_transpose(tensor* data, int is_trans) {
        data->is_trans = is_trans;
    }
    
    inline char get_transpose_char(tensor* data) {
        return data->is_trans ? 't' : 'n';
    }
    
    EXPORT void cuda_sync_threads() {
        cudaDeviceSynchronize();
    }

    EXPORT int allocate_device_memory(tensor* data) {
        int len = 1;
    
        for(int dim = 0; dim < data->dims; dim++){
            len *= data->size[dim];
        }
    
        cudaError_t cuda_status = cudaMalloc((void**)&data->data_device, len * sizeof(data->data_device[0]));
    
        if (cuda_status != cudaSuccess || checkCUDAError()) {
            checkCUDAError();
            return CUDA_ERROR;
        }
    
        data->on_device = 1;
        return 0;
    }
    
    EXPORT int copy_to_host(tensor* data) {
        int len = 1;
    
        for(int dim = 0; dim < data->dims; dim++){
            len *= data->size[dim];
        }
    
        if (data->on_device) {
            cudaError_t cuda_status = cudaMemcpy(data->data_host, data->data_device, len * sizeof(data->data_host[0]), cudaMemcpyDeviceToHost);
    
            if (cuda_status != cudaSuccess || checkCUDAError()) {
                checkCUDAError();
                return CUDA_ERROR;
            }
        } else {
            return ERROR_NOT_ON_DEVICE;
        }
    
     
        return 0;
    }
    
    EXPORT int copy_to_device(tensor* data) {
        int len = 1;
    
        for(int dim = 0; dim < data->dims; dim++){
            len *= data->size[dim];
        }
    
        int err_code = 0;
    

        if (!data->on_device) {
            err_code = allocate_device_memory(data);
            if (err_code)
                return err_code;
        }
    
        cudaError_t cuda_status = cudaMemcpy(data->data_device, data->data_host, len * sizeof(data->data_host[0]), cudaMemcpyHostToDevice);
    
        if (cuda_status != cudaSuccess || checkCUDAError()) {
            checkCUDAError();
            return CUDA_ERROR;
        }

    
        return 0;
    }
    
    EXPORT int copy_on_device(tensor* data1, tensor* data2) {
        int len = 1;
    
        for(int dim = 0; dim < data1->dims; dim++){
            len *= data1->size[dim];
        }
    
        for(int dim = 0; dim < data1->dims; dim++){
            if (data1->size[dim] != data2->size[dim])
                return ERROR_INCOMPATIBLE_DIMENSIONS;
        }
        
        cudaError_t cuda_status = cudaMemcpy(data2->data_device, data1->data_device, len * sizeof(data1->data_device[0]), cudaMemcpyDeviceToDevice);
    
        if (cuda_status != cudaSuccess || checkCUDAError()) {
            checkCUDAError();
            return CUDA_ERROR;
        }
        else
            return 0;
    }
    
    EXPORT int free_device_memory(tensor* data) {
        if (data->on_device) {
            cudaError_t cuda_status = cudaFree(data->data_device);
            data->on_device = 0;
    
            if (cuda_status != cudaSuccess || checkCUDAError()) {
                checkCUDAError();
                return CUDA_ERROR;
            }
        }
    
        return 0;
    }

    EXPORT int build_matrix_empty(int m, int n, tensor* data) {
        data->dims = 2;
        data->size[0] = m;
        data->size[1] = n;
        data->data_host = (float*)malloc(m * n * sizeof(float));
        data->owns_data = 1;
        data->on_device = 0;
        data->on_host = 1;
        return 0;
    }

    EXPORT int build_tensor_empty(int dims, int* size, tensor* data) {
        data->dims = dims;
        for(int i = 0; i < dims; i++){
            data->size[i] = size[i];
        }
        int len = 1;
        for(int i = 0; i < dims; i++){
            len *= size[i];
        }
        data->data_host = (float*)malloc(len * sizeof(float));
        data->owns_data = 1;
        data->on_device = 0;
        data->on_host = 1;
        return 0;
    }


    EXPORT int gemm_on_device(tensor* data1, tensor* data2, tensor* result) {
        int err_code = 0;

        if (data1->on_device || data2->on_device){
            return ERROR_NOT_ON_DEVICE;
        }

        if (data1->dims==2 && data1->dims==2){
            err_code = gemm_on_device_2d(data1, data2, result);
            return err_code;
        }
        else{
            printf("Currently only support matrix multiplication.");
            return ERROR_INCOMLETE;
        }

        if (checkCUDAError())
            return CUDA_ERROR;
        else
            return 0;

    }

    // EXPORT int fft_on_device(tensor* data, tensor* result) {
    //     int err_code = 0;

    //     if (data->on_device){
    //         return ERROR_NOT_ON_DEVICE;
    //     }

    //     if (data->dims==1){
    //         err_code = fft_on_device_1d(data, result);
    //         return err_code;
    //     }
    //     else{
    //         printf("Currently only support 1d fft.");
    //         return ERROR_INCOMLETE;
    //     }

    //     if (checkCUDAError())
    //         return CUDA_ERROR;
    //     else
    //         return 0;

    // }

    EXPORT int fourop_on_device(tensor* data1, tensor* data2, tensor* result, int op) {
        int err_code = 0;
        int eq = 1;

        if (data1->on_device || data2->on_device){
            return ERROR_NOT_ON_DEVICE;
        }

        for(int i = 0; i < data1->dims; i++){
            if (data1->size[i] != data2->size[i]){
                eq = 0;         
            }
        }

        if (eq){
            if (op == 0){
                err_code = add_on_device_eq(data1, data2, result);
            }
            else if (op == 1){
                err_code = sub_on_device_eq(data1, data2, result);
            }
            else if (op == 2){
                err_code = mul_on_device_eq(data1, data2, result);
            }
            else if (op == 3){
                err_code = div_on_device_eq(data1, data2, result);
            }
            else{
                printf("Unsupported operation.");
                return ERROR_UNSUPPORTED;
            }
            return err_code;
        }else{
            printf("Incompatible dimensions.");
            return ERROR_INCOMLETE;
        }

        if (checkCUDAError())
            return CUDA_ERROR;
        else
            return 0;
        
    }

    
    


}