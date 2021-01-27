
#ifndef __CUDAGRADIENTTOTENSOR_3D_FLOAT_VALID_KERNEL__
#define __CUDAGRADIENTTOTENSOR_3D_FLOAT_VALID_KERNEL__

void cudaGradientToTensor_3D_Float_Valid_Kernel(float* d_gx, float* d_gy, float* d_gz,
	int dimx, int dimy, int dimz,
	float* d_smoothmask, int smoothMaskSize,
	float* d_temp1, float* d_temp2,
	float* d_Txx, float* d_Txy, float* d_Txz,
	float* d_Tyy, float* d_Tyz,
	float* d_Tzz);

#endif