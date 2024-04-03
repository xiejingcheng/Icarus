#include <curand.h>
#include "../data_error.cuh"

extern "C"{
    EXPORT void seed(unsigned long long val);
    EXPORT void random_normal(float *a, float mu, float sigma, unsigned int n);
}
