
#include <cudaConvolution_3D_Float_Valid_Kernel.h>
#include <cudaConvolution_2Inputs_3D_Float_Valid_Kernel.h>
#include <cudaGradientToTensor_3D_Float_Valid_Kernel.h>



void cudaGradientToTensor_3D_Float_Valid_Kernel(float* d_gx, float* d_gy, float* d_gz,
	int dimx, int dimy, int dimz,
	float* d_smoothmask, int smoothMaskSize,
	float *d_temp1, float *d_temp2,
	float* d_Txx, float* d_Txy, float* d_Txz,
	float* d_Tyy, float* d_Tyz,
	float* d_Tzz)
{
	int dx = dimx - smoothMaskSize + 1;
	int dy = dimy - smoothMaskSize + 1;
	int dz = dimz - smoothMaskSize + 1;
	cudaConvolution_2Inputs_3D_Float_Valid_X(d_gx, d_gx, dimx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp1);
	cudaConvolution_3D_Float_Valid_Y(d_temp1, dx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp2);
	cudaConvolution_3D_Float_Valid_Z(d_temp2, dx, dy, dimz, d_smoothmask, smoothMaskSize, d_Txx);

	cudaConvolution_2Inputs_3D_Float_Valid_X(d_gx, d_gy, dimx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp1);
	cudaConvolution_3D_Float_Valid_Y(d_temp1, dx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp2);
	cudaConvolution_3D_Float_Valid_Z(d_temp2, dx, dy, dimz, d_smoothmask, smoothMaskSize, d_Txy);

	cudaConvolution_2Inputs_3D_Float_Valid_X(d_gx, d_gz, dimx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp1);
	cudaConvolution_3D_Float_Valid_Y(d_temp1, dx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp2);
	cudaConvolution_3D_Float_Valid_Z(d_temp2, dx, dy, dimz, d_smoothmask, smoothMaskSize, d_Txz);

	cudaConvolution_2Inputs_3D_Float_Valid_X(d_gy, d_gy, dimx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp1);
	cudaConvolution_3D_Float_Valid_Y(d_temp1, dx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp2);
	cudaConvolution_3D_Float_Valid_Z(d_temp2, dx, dy, dimz, d_smoothmask, smoothMaskSize, d_Tyy);

	cudaConvolution_2Inputs_3D_Float_Valid_X(d_gy, d_gz, dimx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp1);
	cudaConvolution_3D_Float_Valid_Y(d_temp1, dx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp2);
	cudaConvolution_3D_Float_Valid_Z(d_temp2, dx, dy, dimz, d_smoothmask, smoothMaskSize, d_Tyz);

	cudaConvolution_2Inputs_3D_Float_Valid_X(d_gz, d_gz, dimx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp1);
	cudaConvolution_3D_Float_Valid_Y(d_temp1, dx, dimy, dimz, d_smoothmask, smoothMaskSize, d_temp2);
	cudaConvolution_3D_Float_Valid_Z(d_temp2, dx, dy, dimz, d_smoothmask, smoothMaskSize, d_Tzz);
}