#include <curand.h>
#include "../data_error.cuh"

extern "C"{
    EXPORT int random_normal(matrix *a, float mu, float sigma, unsigned int n);
}
