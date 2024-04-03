#include "random.cuh"
#include <curand.h>


EXPORT int random_normal(tensor *a, float mu, float sigma, unsigned int n) {
  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT); 
  int len = 1;
  for(int dim = 0; dim < a->dims; dim++){
    len *= a->size[dim];
  }
  curandGenerateUniform(generator, a->data_device, len);
  return 0;
}