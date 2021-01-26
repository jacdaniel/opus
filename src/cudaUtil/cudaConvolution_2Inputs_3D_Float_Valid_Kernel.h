#ifndef __CUDACONVOLUTION_2INPUTS_3D_FLOAT_VALID__
#define __CUDACONVOLUTION_2INPUTS_3D_FLOAT_VALID__


#define BLOCKDIMX 10
#define BLOCKDIMY 10
#define BLOCKDIMZ 10

void cudaConvolution_2Inputs_3D_Float_Valid_X(float* d_in1, float *d_in2, int dimx, int dimy, int dimz, float* d_mask, int maskSize, float* d_out);
void cudaConvolution_2Inputs_3D_Float_Valid_Y(float* d_in1, float *d_in2, int dimx, int dimy, int dimz, float* d_mask, int maskSize, float* d_out);
void cudaConvolution_2Inputs_3D_Float_Valid_Z(float* d_in1, float *d_in2, int dimx, int dimy, int dimz, float* d_mask, int maskSize, float* d_out);


#endif

