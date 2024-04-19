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
#include "operator/non_optimized.cuh"
#include "data_error.cuh"

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

    EXPORT void set_transpose(matrix* data, int is_trans) {
        data->is_trans = is_trans;
    }
    
    inline char get_transpose_char(matrix* data) {
        return data->is_trans ? 't' : 'n';
    }
    
    EXPORT void cuda_sync_threads() {
        cudaDeviceSynchronize();
    }

    EXPORT int allocate_device_memory(matrix* data) {
        int len = data->size[0] * data->size[1];

        cudaError_t cuda_status = cudaMalloc((void**)&data->data_device, len * sizeof(data->data_device[0]));
    
        if (cuda_status != cudaSuccess || checkCUDAError()) {
            checkCUDAError();
            return CUDA_ERROR;
        }
    
        return 0;
    }
    
    EXPORT int copy_to_host(matrix* data) {
        int len = data->size[0] * data->size[1];
    
        if (data->on_device) {
            cudaError_t cuda_status = cudaMemcpy(data->data_host, data->data_device, len * sizeof(data->data_host[0]), cudaMemcpyDeviceToHost);
    
            if (cuda_status != cudaSuccess || checkCUDAError()) {
                checkCUDAError();
                return CUDA_ERROR;
            }
        } else {
            return ERROR_NOT_ON_DEVICE;
        }

        data->on_host = 1;
        data->on_device = 0;
    
     
        return 0;
    }
    
    EXPORT int copy_to_device(matrix* data) {
        int len = data->size[0] * data->size[1];
    
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

        data->on_host = 0;
        data->on_device = 1;
    
        return 0;
    }
    
    EXPORT int copy_on_device(matrix* data1, matrix* data2) {
        if (data1->on_device && data2->on_device)
            return ERROR_NOT_ON_HOST;

        int len = data1->size[0] * data1->size[1];
    
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
    
    EXPORT int free_device_memory(matrix* data) {
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

    EXPORT int build_matrix_empty_host(int m, int n, matrix* data) {
        data->dims = 2;
        data->size[0] = m;
        data->size[1] = n;
        data->data_host = (float*)malloc(m * n * sizeof(float));
        data->owns_data = 1;
        data->on_device = 0;
        data->on_host = 1;
        return 0;
    }

    EXPORT int build_matrix_empty_device(int m, int n, matrix* data) {
        data->dims = 2;
        data->size[0] = m;
        data->size[1] = n;

        data->data_host = (float*)malloc(m * n * sizeof(float));

        int len = data->size[0] * data->size[1];

        cudaError_t cuda_status = cudaMalloc((void**)&data->data_device, len * sizeof(data->data_device[0]));
    
        if (cuda_status != cudaSuccess || checkCUDAError()) {
            checkCUDAError();
            return CUDA_ERROR;
        }
        data->owns_data = 1;
        data->on_host = 0;
        data->on_device = 1;
        return 0;
    }

    EXPORT int build_matrix_from_array(int m, int n, matrix *mat, float *arr)
    {
        mat->dims = 2;
        mat->size[0] = m;
        mat->size[1] = n;
        mat->data_host = arr;
        mat->owns_data = 0;
        mat->on_device = 0;
        mat->on_host = 1;
        return 0;
    }

    EXPORT int from_matrix_to_array(matrix *mat, float *arr)
    {
        int len = mat->size[0] * mat->size[1];

        if (mat->on_device)
        {   
            int err_code = copy_to_host(mat);
            if (err_code)
                return err_code;
        }

        for (int i = 0; i < len; i++)
        {
            arr[i] = mat->data_host[i];
        }
        return 0;
    }

    EXPORT int gemm_on_device(matrix* data1, matrix* data2, matrix* result) {
        int err_code = 0;

        if (!data1->on_device || !data2->on_device){
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

    EXPORT int fourop_on_device(matrix* data1, matrix* data2, matrix* result, int op) {
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