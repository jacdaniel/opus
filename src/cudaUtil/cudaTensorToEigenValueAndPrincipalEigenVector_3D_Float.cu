
#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <eigenUtil.h>
#include <cudaTensorToEigenValueAndPrincipalEigenVector_3D_Float.h>


__global__ void cudaTensorToEigenValueAndPrincipalEigenVector_3D_Float_Kernel(
	float* Txx, float* Txy, float* Txz, float* Tyy, float* Tyz, float* Tzz,
	long size, 
	float *lambda1, float *lambda2, float *lambda3,
	float* nx, float* ny, float* nz)
{
	const long add = blockIdx.x * blockDim.x + threadIdx.x;

	if (add < size)
	{
		// EIGEN_PRINCIPAL_VECTOR_SDP3X3_F(Txx[add], Tyy[add], Tzz[add], Txy[add], Txz[add], Tyz[add], nx[add], ny[add], nz[add])
		SPD_3X3_EIGEN_VALUES_AND_PRINCIPAL_EIGEN_VECTOR(Txx[add], Tyy[add], Tzz[add], Txy[add], Txz[add], Tyz[add], lambda1[add], lambda2[add], lambda3[add], nx[add], ny[add], nz[add])

	}
}



void cudaTensorToEigenValueAndPrincipalEigenVector_3D_Float(float* d_Txx, float* d_Txy, float* d_Txz, float* d_Tyy, float* d_Tyz, float* d_Tzz,
	long size, float *d_lambda1, float *d_lambda2, float *d_lambda3, float* d_nx, float* d_ny, float* d_nz)
{
	dim3 block(1024);
	dim3 grid((size - 1) / block.x + 1);
	cudaTensorToEigenValueAndPrincipalEigenVector_3D_Float_Kernel << <grid, block >> > (d_Txx, d_Txy, d_Txz, d_Tyy, d_Tyz, d_Tzz, size, d_lambda1, d_lambda2, d_lambda3, d_nx, d_ny, d_nz);
	cudaThreadSynchronize();
}