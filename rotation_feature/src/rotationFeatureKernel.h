typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

#define NUM_ROT 4

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
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
                     const uint32 count);

void rotationGrad(cudaStream_t stream,
                    double* gradData,
                    double* outputData,
                    const uint16 batch,
                    const uint16 channel,
                    const uint16 H,
                    const uint32 count);


#ifdef __cplusplus
}
#endif