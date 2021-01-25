
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaConvolution_3D_Float_Valid_Kernel.h>

// ========================================================
// KERNEL
// ========================================================

__global__ void cudaConvolution_3D_Float_Valid_X_Kernel(float* in, int dimx, int dimy, int dimz, float* mask, int size2, float* out)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x,
		y = blockIdx.y * blockDim.y + threadIdx.y,
		z = blockIdx.z * blockDim.z + threadIdx.z;

	// in += baseZ * (dimx * dimy) + baseY * dimx + baseX;
	// out += baseZ * ((dimx - 2 * size2) * dimy) + baseY * (dimx - 2 * size2) + baseX - size2;

	if (x >= size2 && x < dimx-size2 && y < dimy && z < dimz)
	{
		float sum = 0.0f;
#pragma unroll
			for (int i = -size2; i <= size2; i++)
				sum += mask[size2 - i] * in[(long)dimx*dimy* z+(long)dimx*y+(long)x+i];
			out[(long)z * ((dimx - 2 * size2) * dimy) + (long)y * (dimx - 2 * size2) + (long)x-size2] = sum;
	}
}



__global__ void cudaConvolution_3D_Float_Valid_Y_Kernel(float* in, int dimx, int dimy, int dimz, float* mask, int size2, float* out)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x,
		y = blockIdx.y * blockDim.y + threadIdx.y,
		z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < dimx && y >= size2 && y < dimy - size2 && z < dimz)
	{
		float sum = 0.0f;
#pragma unroll
		for (int i = -size2; i <= size2; i++)
			sum += mask[size2 - i] * in[dimx * dimy * z + dimx * (y+i) + x];
		out[z * ((dimx - 2 * size2) * dimy) + (y-size2) * (dimx - 2 * size2) + x] = sum;
	}
}


__global__ void cudaConvolution_3D_Float_Valid_Z_Kernel(float* in, int dimx, int dimy, int dimz, float* mask, int size2, float* out)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x,
		y = blockIdx.y * blockDim.y + threadIdx.y,
		z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < dimx && y < dimy && z >= size2 && z < dimz - size2)
	{
		float sum = 0.0f;
#pragma unroll
		for (int i = -size2; i <= size2; i++)
			sum += mask[size2 - i] * in[dimx * dimy * (z+i) + dimx * y + x];
		out[(z-size2) * ((dimx - 2 * size2) * dimy) + y * (dimx - 2 * size2) + x] = sum;
	}
}




void cudaConvolution_3D_Float_Valid_X(float* d_in, int dimx, int dimy, int dimz, float* d_mask, int maskSize, float* d_out)
{
	dim3 block(10, 10, 10);
	dim3 grid((dimx - 1) / block.x + 1, (dimy - 1) / block.y + 1, (dimz - 1) / block.z + 1);
	cudaConvolution_3D_Float_Valid_X_Kernel << <grid, block >> > (d_in, dimx, dimy, dimz, d_mask, maskSize / 2, d_out);
	cudaThreadSynchronize();
}

void cudaConvolution_3D_Float_Valid_Y(float* d_in, int dimx, int dimy, int dimz, float* d_mask, int maskSize, float* d_out)
{
	dim3 block(10, 10, 10);
	dim3 grid((dimx - 1) / block.x + 1, (dimy - 1) / block.y + 1, (dimz - 1) / block.z + 1);
	cudaConvolution_3D_Float_Valid_Y_Kernel << <grid, block >> > (d_in, dimx, dimy, dimz, d_mask, maskSize / 2, d_out);
	cudaThreadSynchronize();
}

void cudaConvolution_3D_Float_Valid_Z(float* d_in, int dimx, int dimy, int dimz, float* d_mask, int maskSize, float* d_out)
{
	dim3 block(10, 10, 10);
	dim3 grid((dimx - 1) / block.x + 1, (dimy - 1) / block.y + 1, (dimz - 1) / block.z + 1);
	cudaConvolution_3D_Float_Valid_Z_Kernel << <grid, block >> > (d_in, dimx, dimy, dimz, d_mask, maskSize / 2, d_out);
	cudaThreadSynchronize();
}

// #define CUDA_MEM_CPY_TO_SYMBOL_FLOAT(_dst, _src, _size) cudaMemcpyToSymbol(_dst, _src, _size*sizeof(float));




/*

__global__ void cudaConvolution3DValidXKernel(float* in, int dimx, int dimy, int dimz, float *mask, int size2, float* out)
{
	__shared__ float s_Data[BLOCKDIMZ][BLOCKDIMY][BLOCKDIMX * 3];
	const int baseX = (blockIdx.x - 1) * blockDim.x + threadIdx.x,
		baseY = blockIdx.y * blockDim.y + threadIdx.y,
		baseZ = blockIdx.z * blockDim.z + threadIdx.z;

	in += baseZ * (dimx * dimy) + baseY * dimx + baseX;
	out += baseZ * ((dimx - 2 * size2) * dimy) + baseY * (dimx - 2 * size2) + baseX - size2;

	if (baseX < dimx && baseY < dimy && baseZ < dimz)
	{
		s_Data[threadIdx.z][threadIdx.y][threadIdx.x + BLOCKDIMX] = (baseX + BLOCKDIMX < dimx) ? in[BLOCKDIMX] : 0.0f;
		s_Data[threadIdx.z][threadIdx.y][threadIdx.x] = (baseX > 0) ? in[0] : 0.0f;
		s_Data[threadIdx.z][threadIdx.y][threadIdx.x + 2 * BLOCKDIMX] = (dimx - baseX > 2 * BLOCKDIMX) ? in[2 * BLOCKDIMX] : 0.0f;
		__syncthreads();

		if (baseX + BLOCKDIMX >= size2 && baseX + BLOCKDIMX + size2 < dimx)
		{
			float sum = 0.0f;

#pragma unroll
			for (int i = -size2; i <= size2; i++)
				sum += mask[size2 - i] * s_Data[threadIdx.z][threadIdx.y][threadIdx.x + BLOCKDIMX + i];
			out[BLOCKDIMX] = sum;
		}
	}
}

*/