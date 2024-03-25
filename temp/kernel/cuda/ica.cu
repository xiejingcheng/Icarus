#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils/device.cuh"
#include "ica.cuh"

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
        cudaThreadSynchronize();
    }



}