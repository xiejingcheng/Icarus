#include "random.cuh"
#include <curand.h>


EXPORT int random_normal(matrix *a, float mu, float sigma, unsigned int n) {
  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT); 
  int len = a->size[0] * a->size[1];
  curandGenerateUniform(generator, a->data_device, len);
  return 0;
}