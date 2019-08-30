#include<stdio.h>
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/rotationFeatureCuda.cu"
#else

void rotation_feature(THCTensor *feature,
                      THCTensor *output){

    THCUNN_assertSameGPU(state, 2, feature, output);

    const uint16 batch = feature->size[0];
    const uint16 channel = feature->size[1];
    const uint16 H = feature->size[2];
    const uint16 W = feature->size[3];

    const uint32 count = batch*channel*H*W;

    THCTensor_(resize4d)(state, output, batch, channel, H, W);
    real *featureData = THCTensor_(data)(state, feature);
    real *outputData = THCTensor_(data)(state, output);

    rotationFeature(THCState_getCurrentStream(state),
                    featureData,
                    outputData,
                    batch,
                    channel,
                    H,
                    count);
    THCudaCheck(cudaGetLastError());
}

void rotation_grad(THCTensor *grad,
                   THCTensor *output){
    THCUNN_assertSameGPU(state, 2, grad, output);

    const uint16 batch = grad->size[0];
    const uint16 channel = grad->size[1];
    const uint16 H = grad->size[2];
    const uint16 W = grad->size[3];
    const uint32 count = batch*channel*H*W;

    THCTensor_(resize4d)(state, output, batch, channel, H, W);
    real *gradData = THCTensor_(data)(state, grad);
    real *outputData = THCTensor_(data)(state, output);

    rotationGrad(THCState_getCurrentStream(state),
                    gradData,
                    outputData,
                    batch,
                    channel,
                    H,
                    count);
    THCudaCheck(cudaGetLastError());
}

#endif