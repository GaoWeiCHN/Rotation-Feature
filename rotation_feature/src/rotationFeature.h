typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

void rotation_feature(THCudaDoubleTensor *feature,
                      THCudaDoubleTensor *output);

void rotation_grad(THCudaDoubleTensor *grad,
                   THCudaDoubleTensor *output);