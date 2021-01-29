#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <cuda_runtime.h>

#include <util0.h>
#include <gaussian.hpp>
#include <cudaGradient_3D_Float_Valid_Kernel.h>
#include <cudaGradientToTensor_3D_Float_Valid_Kernel.h>
#include <cudaTensorToEigenValueAndPrincipalEigenVector_3D_Float.h>
#include <cudaTensorToPrincipalEigenVector_3D_Float.h>

#define PI 3.1415926

void dataRead(char* filename, int x1, int x2, int y1, int y2, int z1, int z2, float* out);

void syntheticDataCreate_v1(int dimx, int dimy, int dimz, float* data)
{
	float fx = 10.0f;
	float py = 2.f, pz = .5f;
	for (long z = 0; z < dimz; z++)
	{
		double zz = (double)z / dimz;
		for (long y = 0; y < dimy; y++)
		{
			double yy = (double)y / dimy;
			for (long x = 0; x < dimx; x++)
			{
				double xx = (double)x / dimx;
				py = sin(2*PI*yy);
				double arg = fx * (xx - py * yy - pz * zz);
				data[dimx * dimy * z + dimx * y + x] = 255.0f * sin(2.0 * PI * arg);
			}
		}
	}
}

int main_cudaGradient(int argc, char** argv)
{
	char* seismicFilename = "D:\\TOTAL\\PLI\\seismic.xt";

	int dimx = 101, dimy = 101, dimz = 101;
	long size0 = (long)dimx * dimy * dimz;
	float* in = (float*)calloc(size0, sizeof(float));
	float* d_in = nullptr, * d_tmp1 = nullptr, * d_tmp2 = nullptr, * d_gx = nullptr, * d_gy = nullptr, * d_gz = nullptr, * d_lpmask = nullptr, * d_hpmask = nullptr;
	float* lpmask = nullptr, * hpmask = nullptr;
	float* d_Txx = nullptr, * d_Txy = nullptr, * d_Txz = nullptr, * d_Tyy = nullptr, * d_Tyz = nullptr, * d_Tzz = nullptr;
	long lpmaskSize, hpmaskSize;
	double sigma = 1.0;

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
	cudaMalloc((void**)&d_Txx, size0 * sizeof(float));
	cudaMalloc((void**)&d_Txy, size0 * sizeof(float));
	cudaMalloc((void**)&d_Txz, size0 * sizeof(float));
	cudaMalloc((void**)&d_Tyy, size0 * sizeof(float));
	cudaMalloc((void**)&d_Tyz, size0 * sizeof(float));
	cudaMalloc((void**)&d_Tzz, size0 * sizeof(float));

	// in[dimx * dimy * 50 + dimx * 50 + 50] = 1.0;
	dataRead(seismicFilename, 0, dimx - 1, 0, dimy - 1, 0, dimz - 1, in);
	syntheticDataCreate_v1(dimx, dimy, dimz, in);
	cudaMemcpy(d_in, in, size0 * sizeof(float), cudaMemcpyHostToDevice);

	FILE* pFile = fopen("d:\\in.raw", "wb");
	fwrite(in, sizeof(float), (size_t)dimx * dimy * dimz, pFile);
	fclose(pFile);

	int dx = dimx - lpmaskSize + 1;
	int dy = dimy - lpmaskSize + 1;
	int dz = dimz - lpmaskSize + 1;

	cudaGradient_3D_Float_Valid(d_in, dimx, dimy, dimz, d_lpmask, lpmaskSize, d_hpmask, hpmaskSize, d_tmp1, d_tmp2, d_gx, d_gy, d_gz);

	// cudaMemcpy(in, d_tmp1, dimx * dimy * dz * sizeof(float), cudaMemcpyDeviceToHost);
	// FILE* pFile = fopen("d:\\gx.raw", "wb");
	// fwrite(in, sizeof(float), dimx * dimy * dz, pFile);
	// fclose(pFile);


	cudaMemcpy(in, d_gx, (size_t)dx * dy * dz * sizeof(float), cudaMemcpyDeviceToHost);
	pFile = fopen("d:\\gx.raw", "wb");
	fwrite(in, sizeof(float), (size_t)dx * dy * dz, pFile);
	fclose(pFile);

	cudaMemcpy(in, d_gy, (size_t)dx * dy * dz * sizeof(float), cudaMemcpyDeviceToHost);
	pFile = fopen("d:\\gy.raw", "wb");
	fwrite(in, sizeof(float), (size_t)dx * dy * dz, pFile);
	fclose(pFile);

	cudaMemcpy(in, d_gz, (size_t)dx * dy * dz * sizeof(float), cudaMemcpyDeviceToHost);
	pFile = fopen("d:\\gz.raw", "wb");
	fwrite(in, sizeof(float), (size_t)dx * dy * dz, pFile);
	fclose(pFile);

	// tensor
	cudaGradientToTensor_3D_Float_Valid_Kernel(d_gx, d_gy, d_gz, dx, dy, dz,
		d_lpmask, lpmaskSize,
		d_tmp1, d_tmp2,
		d_Txx, d_Txy, d_Txz,
		d_Tyy, d_Tyz,
		d_Tzz);

	int ddx = dx - lpmaskSize + 1;
	int ddy = dy - lpmaskSize + 1;
	int ddz = dz - lpmaskSize + 1;

	cudaMemcpy(in, d_Tzz, (size_t)ddx * ddy * ddz * sizeof(float), cudaMemcpyDeviceToHost);
	pFile = fopen("d:\\Tzz.raw", "wb");
	fwrite(in, sizeof(float), (size_t)ddx * ddy * ddz, pFile);
	fclose(pFile);

	// ===========================================================================
	float* d_nx = nullptr, * d_ny = nullptr, * d_nz = nullptr;
	long size2 = (long)ddx * ddy * ddz;

	int ret0 = CUDAMALLOCSAFE(&d_nx, size2 * sizeof(float));
	ret0 = CUDAMALLOCSAFE(&d_ny, size2 * sizeof(float));
	ret0 = CUDAMALLOCSAFE(&d_nz, size2 * sizeof(float));

	cudaTensorToPrincipalEigenVector_3D_Float(d_Txx, d_Txy, d_Txz, d_Tyy, d_Tyz, d_Tzz, (long)ddx*ddy*ddz, d_nx, d_ny, d_nz);

	float* nx = nullptr, * ny = nullptr, * nz = nullptr;;
	int ret = CALLOCSAFE(&nx, (size_t)size2, float);
	ret = CALLOCSAFE(&ny, (size_t)size2, float);
	ret = CALLOCSAFE(&nz, (size_t)size2, float);

	cudaMemcpy(nx, d_nx, size2 * sizeof(float), cudaMemcpyDeviceToHost);
	float nnx = nx[ddx * ddy * (ddz / 2) + ddx * (ddy / 2) + ddx / 2];
	pFile = fopen("d:\\nx.raw", "wb");
	fwrite(nx, sizeof(float), (size_t)ddx * ddy * ddz, pFile);
	fclose(pFile);

	cudaMemcpy(ny, d_ny, size2 * sizeof(float), cudaMemcpyDeviceToHost);
	float nny = ny[ddx * ddy * (ddz / 2) + ddx * (ddy / 2) + ddx / 2];
	pFile = fopen("d:\\ny.raw", "wb");
	fwrite(ny, sizeof(float), (size_t)ddx * ddy * ddz, pFile);
	fclose(pFile);

	cudaMemcpy(nz, d_nz, size2 * sizeof(float), cudaMemcpyDeviceToHost);
	float nnz = nz[ddx * ddy * (ddz / 2) + ddx * (ddy / 2) + ddx / 2];
	pFile = fopen("d:\\nz.raw", "wb");
	fwrite(nz, sizeof(float), (size_t)ddx* ddy* ddz, pFile);
	fclose(pFile);

	float pxy = -nny / nnx;
	float pxz = -nnz / nnx;

	fprintf(stderr, "dipx: %f\ndipz: %f\n", pxy, pxz);


	for (long n = 0; n < size2; n++)
	{
		ny[n] /= -nx[n];
		nz[n] /= -nx[n];
	}

	pFile = fopen("d:\\dipx.raw", "wb");
	fwrite(ny, sizeof(float), (size_t)ddx * ddy * ddz, pFile);
	fclose(pFile);

	pFile = fopen("d:\\dipz.raw", "wb");
	fwrite(nz, sizeof(float), (size_t)ddx * ddy * ddz, pFile);
	fclose(pFile);


	
	FREESAFE(nx)
	CUDAFREESAFE(&d_nx)
	return 0;
}