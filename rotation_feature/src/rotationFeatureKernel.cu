#include"rotationFeatureKernel.h"
#include<stdio.h>

__global__ void rotationFeatureKernel(double* featureData,
                                        double* outputData,
                                        const uint16 batch,
                                        const uint16 channel,
                                        const uint16 H){
    const uint32 threadID = blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
    const uint16 channelNum = threadID / (H*H);
    const uint16 channelNo = channelNum % channel;
    const uint16 elementNum = threadID % (H*H);
    const uint16 rowNum = elementNum / H;
    const uint16 colNum = elementNum % H;

    if(channelNo == 1)
        *(outputData + channelNum*H*H + H*colNum+H-rowNum-1) = *(featureData+threadID);
    else if(channelNo == 2)
        *(outputData + channelNum*H*H + (H-rowNum-1)*H + H-colNum-1) = *(featureData+threadID);
    else if(channelNo == 3)
        *(outputData + threadID) = *(featureData + channelNum*H*H +  H*colNum+H-rowNum-1);
    else
        *(outputData+threadID) = *(featureData+threadID);
}

__global__ void rotationGradKernel(double* gradData,
                                   double* outputData,
                                   const uint16 batch,
                                   const uint16 channel,
                                   const uint16 H){
    const uint32 threadID = blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
    const uint16 channelNum = threadID / (H*H);
    const uint16 channelNo = channelNum % channel;
    const uint16 elementNum = threadID % (H*H);
    const uint16 rowNum = elementNum / H;
    const uint16 colNum = elementNum % H;

    if(channelNo == 1)
        *(outputData + threadID) = *(gradData + channelNum*H*H +  H*colNum+H-rowNum-1);
    else if(channelNo == 2)
        *(outputData + channelNum*H*H + (H-rowNum-1)*H + H-colNum-1) = *(gradData+threadID);
    else if(channelNo == 3)
        *(outputData + channelNum*H*H + H*colNum+H-rowNum-1) = *(gradData+threadID);
    else
        *(outputData+threadID) = *(gradData+threadID);
}

#ifdef __cplusplus
extern "C" {
#endif

void rotationFeature(cudaStream_t stream,
                     double* featureData,
                     double* outputData,
                     const uint16 batch,
                     const uint16 channel,
                     const uint16 H,
                     const uint32 count){
    rotationFeatureKernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0>>>(featureData, outputData, batch, channel, H);
}

void rotationGrad(cudaStream_t stream,
                    double* gradData,
                    double* outputData,
                    const uint16 batch,
                    const uint16 channel,
                    const uint16 H,
                    const uint32 count){
    rotationGradKernel<<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0>>>(gradData, outputData, batch, channel, H);
}

#ifdef __cplusplus
}
#endif