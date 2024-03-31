#include "../data_error.cuh"

extern "C" {
    EXPORT int fft_on_device_1d(tensor* data, tensor* result);
    
};