#include "../data_error.cuh"

extern "C" {
    EXPORT int add_on_device_eq(tensor* data1, tensor* data2, tensor* result);
    EXPORT int sub_on_device_eq(tensor* data1, tensor* data2, tensor* result);
    EXPORT int mul_on_device_eq(tensor* data1, tensor* data2, tensor* result);
    EXPORT int div_on_device_eq(tensor* data1, tensor* data2, tensor* result);  
};