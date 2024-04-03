#include "../data_error.cuh"

extern "C"{
    EXPORT int save_tensor(tensor* data, const char* filename);
}