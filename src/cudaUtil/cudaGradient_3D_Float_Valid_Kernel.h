#ifndef __CUDAGRADIENT_3D_FLOAT_VALID_KERNEL__
#define __CUDAGRADIENT_3D_FLOAT_VALID_KERNEL__


void cudaGradient_3D_Float_Valid(float* d_in, int dimx, int dimy, int dimz, float* d_LPMask, int LPMaskSize, float* d_HPMask, int HPMaskSize,
	float* d_tmp, float* d_tmp2, float* d_gx, float* d_gy, float* d_gz);


#endif