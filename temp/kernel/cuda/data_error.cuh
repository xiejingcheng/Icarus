#include <cuda_runtime.h>
#include <errno.h>

#ifndef DATA_ERROR_CUH_INCLUDED
#define DATA_ERROR_CUH_INCLUDED

// 设置可见性
#if defined(_WIN32) || defined(__CYGWIN__)
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__ ((visibility("default")))
#endif

// 定义错误码
#define SYNC_THREADS 1

#define ERROR_INCOMPATIBLE_DIMENSIONS -1
#define CUBLAS_ERROR -2
#define CUDA_ERROR -3
#define VIEW_ERROR -4
#define ERROR_TRANSPOSED -5
#define ERROR_GENERIC -6
#define ERROR_TRANSPOSEDNESS -7
#define ERROR_NOT_ON_DEVICE -8
#define ERROR_UNSUPPORTED -9
#define ERROR_FILE_OPEN -10
#define ERROR_FILE_SCAN -11
#define ERROR_NOT_ON_HOST -12

#define ERROR_INCOMLETE -99



struct matrix {
    float* data_host;
    float* data_device;
    int on_device = 0;
    int on_host = 1;
    int dims;
    int size[2];
    int is_trans;
    int owns_data;
};

struct rnd_struct {
    unsigned int* dev_mults;
    unsigned long long* dev_words;
};


#endif 