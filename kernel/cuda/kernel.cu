#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include "kernels.cuh"

extern "C" {

//--------------------------检查cuda cublas 是否工作----------------------

//内联检查函数 检查cublas cuda错误
inline bool check_cublas_error() {
    cublasStatus status = cublasGetError();

    return status != CUBLAS_STATUS_SUCCESS;
}

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

//可以从动态链接库中调出（给python用） 获取cuda的错误
EXPORT const char* get_last_cuda_error() {
    cudaError_t err = cudaGetLastError();

    return cudaGetErrorString( err);
}

EXPORT const char* get_last_clib_error() {
    return strerror(errno);
}

EXPORT int cublas_init() {
    cublasInit();
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

//用于关闭 cuBLAS 库和 CUDA 线程
EXPORT int cublas_shutdown() {
    cublasShutdown();
    cudaThreadExit();

    return 0;
}

EXPORT int cuda_set_device(int deviceId) {
    cudaSetDevice(deviceId);
    
    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

EXPORT int init_random(rnd_struct* rnd_state, int seed, char* cudamatpath) {
    unsigned int * host_mults;
    host_mults = (unsigned int*)malloc(NUM_RND_STREAMS * sizeof(unsigned int));
    FILE * pFile;

    pFile = fopen (cudamatpath,"r");
    if (pFile == NULL) {
        return ERROR_FILE_OPEN;
    }

    for (int i = 0; i < NUM_RND_STREAMS; i++) {
        if (fscanf (pFile, "%u", &host_mults[i]) != 1) {
            return ERROR_FILE_SCAN;
        }
    }
    fclose (pFile);

    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned int), (void**)&rnd_state->dev_mults);
    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned long long), (void**)&rnd_state->dev_words);
    cublasSetVector(NUM_RND_STREAMS, sizeof(unsigned int), host_mults, 1, rnd_state->dev_mults, 1);
    //cudaMalloc((void **)&rnd_state->dev_mults, NUM_RND_STREAMS * sizeof(unsigned int));
    //cudaMalloc((void **)&rnd_state->dev_words, NUM_RND_STREAMS * sizeof(unsigned long long));
    //cudaMemcpy(rnd_state->dev_mults, host_mults, NUM_RND_STREAMS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    kSeedRandom<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, seed);
 
    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

//---------------------------------一些工具函数-------------------------
//tensor不合适
// EXPORT int get_leading_dimension(tensor* mat) {
//     return mat->is_trans ? mat->size[1] : mat->size[0];
// }

// EXPORT int get_nonleading_dimension(tensor* mat) {
//     return mat->is_trans ? mat->size[0] : mat->size[1];
// }


EXPORT void set_transpose(tensor* data, int is_trans) {
    data->is_trans = is_trans;
}

inline char get_transpose_char(tensor* data) {
    return data->is_trans ? 't' : 'n';
}

EXPORT void cuda_sync_threads() {
    cudaThreadSynchronize();
}

//-----------------------数据在设备和主机迁移-----------

EXPORT int allocate_device_memory(tensor* data) {
    int len = 1;

    for(int dim = 0; dim < data->dims; dim++){
        len *= data->size[dim];
    }

    cublasStatus stat;

    stat = cublasAlloc(len, sizeof(data->data_device[0]), (void**)&data->data_device);

    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error()) {
        checkCUDAError();
        return CUBLAS_ERROR;
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
            cublasGetVector(len, sizeof(data->data_host[0]), data->data_device, 1, data->data_host, 1);

        if (check_cublas_error())
            return CUBLAS_ERROR;
    } else
       return ERROR_NOT_ON_DEVICE;
 
    return 0;
}

EXPORT int copy_to_device(tensor* data) {
    int len = 1;

    for(int dim = 0; dim < data->dims; dim++){
        len *= data->size[dim];
    }

    int err_code = 0;

    //if (!mat->owns_data)
    //    return VIEW_ERROR;

    if (!data->on_device) {
        err_code = allocate_device_memory(data);
        if (err_code)
            return err_code;
    }

    cublasSetVector(len, sizeof(data->data_host[0]), data->data_host, 1, data->data_device, 1);
    
    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

EXPORT int copy_on_device(tensor* data1, tensor* data2) {
    int len = 1;

    for(int dim = 0; dim < data->dims; dim++){
        len *= data->size[dim];
    }

    for(int dim = 0; dim < data->dims; dim++){
        if (mat1->size[dim] != mat2->size[dim])
            return ERROR_INCOMPATIBLE_DIMENSIONS;
    }
    
    cublasScopy(len, data1->data_device, 1, data2->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}



}
