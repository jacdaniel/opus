#include <stdio.h>
#include <malloc.h>
#include <cuda_runtime.h>

#include <gaussian.hpp>
#include <cudaGradient_3D_Float_Valid_Kernel.h>


int main_cudaGradient(int argc, char** argv)
{
	int dimx = 101, dimy = 101, dimz = 101;
	long size0 = (long)dimx * dimy * dimz;
	float* in = (float*)calloc(size0, sizeof(float));
	float* d_in = nullptr, * d_tmp1 = nullptr, * d_tmp2 = nullptr, * d_gx = nullptr, * d_gy = nullptr, * d_gz = nullptr, * d_lpmask = nullptr, * d_hpmask = nullptr;
	float* lpmask = nullptr, * hpmask = nullptr;
	long lpmaskSize, hpmaskSize, sigma = 1.0;

	lpmask = gaussianMask1D<float>(sigma, &lpmaskSize);
	hpmask = gaussianGradMask1D<float>(sigma, &hpmaskSize);

	cudaMalloc((void**)&d_lpmask, size0 * sizeof(float));
	cudaMalloc((void**)&d_hpmask, size0 * sizeof(float));
	cudaMemcpy(d_lpmask, lpmask, lpmaskSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hpmask, hpmask, hpmaskSize * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_in, size0 * sizeof(float));
	cudaMalloc((void**)&d_tmp1, size0 * sizeof(float));
	cudaMalloc((void**)&d_tmp2, size0 * sizeof(float));
	cudaMalloc((void**)&d_gx, size0 * sizeof(float));
	cudaMalloc((void**)&d_gy, size0 * sizeof(float));
	cudaMalloc((void**)&d_gz, size0 * sizeof(float));

	in[dimx * dimy * 50 + dimx * 50 + 50] = 1.0;
	cudaMemcpy(d_in, in, size0 * sizeof(float), cudaMemcpyHostToDevice);

	int dx = dimx - lpmaskSize + 1;
	int dy = dimy - lpmaskSize + 1;
	int dz = dimz - lpmaskSize + 1;

	cudaGradient_3D_Float_Valid(d_in, dimx, dimy, dimz, d_lpmask, lpmaskSize, d_hpmask, hpmaskSize, d_tmp1, d_tmp2, d_gx, d_gy, d_gz);

	// cudaMemcpy(in, d_tmp1, dimx * dimy * dz * sizeof(float), cudaMemcpyDeviceToHost);
	// FILE* pFile = fopen("d:\\gx.raw", "wb");
	// fwrite(in, sizeof(float), dimx * dimy * dz, pFile);
	// fclose(pFile);


	cudaMemcpy(in, d_gx, dx * dy * dz * sizeof(float), cudaMemcpyDeviceToHost);
	FILE* pFile = fopen("d:\\gx.raw", "wb");
	fwrite(in, sizeof(float), dx * dy * dz, pFile);
	fclose(pFile);

	cudaMemcpy(in, d_gy, dx * dy * dz * sizeof(float), cudaMemcpyDeviceToHost);
	pFile = fopen("d:\\gy.raw", "wb");
	fwrite(in, sizeof(float), dx * dy * dz, pFile);
	fclose(pFile);

	cudaMemcpy(in, d_gz, dx * dy * dz * sizeof(float), cudaMemcpyDeviceToHost);
	pFile = fopen("d:\\gz.raw", "wb");
	fwrite(in, sizeof(float), dx * dy * dz, pFile);
	fclose(pFile);

	return 0;
}