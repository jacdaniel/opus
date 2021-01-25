#ifndef __CUDACONVOLUTION_SHARED_MEMORY_3D_FLOAT_VALID__
#define __CUDACONVOLUTION_SHARED_MEMORY_3D_FLOAT_VALID__


#define BLOCKDIMX 10
#define BLOCKDIMY 10
#define BLOCKDIMZ 10

void cudaConvolution_Shared_Memory_3D_Float_Valid_X(float* d_in, int dimx, int dimy, int dimz, float* mask, int maskSize, float* d_out);


#endif
