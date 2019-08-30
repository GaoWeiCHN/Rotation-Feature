#include <THC/THC.h>
#include "rotationFeatureKernel.h"
#include "rotationFeature.h"

#define THCUNN_assertSameGPU(...) THAssertMsg(THCudaTensor_checkGPU(__VA_ARGS__), \
  "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")

extern THCState *state;

//#include "generic/featureRotation_cuda.cu"
//#include <THC/THCGenerateFloatType.h>
//
#include "generic/rotationFeatureCuda.cu"
#include <THC/THCGenerateDoubleType.h>