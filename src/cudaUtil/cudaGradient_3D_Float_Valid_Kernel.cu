
#include <cuda_runtime.h>

#include <cudaConvolution_3D_Float_Valid_Kernel.h>
#include <cudaGradient_3D_Float_Valid_Kernel.h>

void cudaGradient_3D_Float_Valid(float* d_in, int dimx, int dimy, int dimz, float* d_LPMask, int LPMaskSize,  float *d_HPMask, int HPMaskSize, 
	float *d_tmp, float *d_tmp2, float* d_gx, float *d_gy, float *d_gz)
{
	int dx = dimx - LPMaskSize + 1, dy = dimy - LPMaskSize + 1, dz = dimz - LPMaskSize + 1;

	cudaConvolution_3D_Float_Valid_Z(d_in, dimx, dimy, dimz,  d_LPMask, LPMaskSize, d_tmp);

	cudaConvolution_3D_Float_Valid_Y(d_tmp, dimx, dimy, dz, d_LPMask, LPMaskSize, d_tmp2);
	cudaConvolution_3D_Float_Valid_X(d_tmp2, dimx, dy, dz, d_HPMask, HPMaskSize, d_gx);

	cudaConvolution_3D_Float_Valid_X(d_tmp, dimx, dimy, dz, d_LPMask, LPMaskSize, d_tmp2);
	cudaConvolution_3D_Float_Valid_Y(d_tmp2, dx, dimy, dz, d_HPMask, HPMaskSize, d_gy);

	cudaConvolution_3D_Float_Valid_X(d_in, dimx, dimy, dimz, d_LPMask, LPMaskSize, d_tmp);
	cudaConvolution_3D_Float_Valid_Y(d_tmp, dx, dimy, dimz, d_LPMask, LPMaskSize, d_tmp2);
	cudaConvolution_3D_Float_Valid_Z(d_tmp2, dx, dy, dimz, d_HPMask, HPMaskSize, d_gz);
}