
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaConvolution_Shared_Memory_3D_Float_Valid_Kernel.h>

// ========================================================
// KERNEL
// ========================================================

__global__ void cudaConvolution_Shared_Memory_3D_Float_Valid_X_Kernel(float* in, int dimx, int dimy, int dimz, float *mask, int size2, float* out)
{

	__shared__ float s_Data[BLOCKDIMZ][BLOCKDIMY][BLOCKDIMX * 3];
	const int baseX = (blockIdx.x - 1) * blockDim.x + threadIdx.x,
		baseY = blockIdx.y * blockDim.y + threadIdx.y,
		baseZ = blockIdx.z * blockDim.z + threadIdx.z;

	in += baseZ * (dimx * dimy) + baseY * dimx + baseX;
	out += baseZ * ((dimx - 2 * size2) * dimy) + baseY * (dimx - 2 * size2) + baseX - size2;
	// out += baseZ * ((dimx) * dimy) + baseY * (dimx) + baseX;

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



__global__ void cudaConvolution_Shared_Memory_3D_Float_Valid_Y_Kernel(float* in, int dimx, int dimy, int dimz, float *mask, int size2, float* out)
{
	__shared__ float s_Data[BLOCKDIMZ][BLOCKDIMX][BLOCKDIMY * 3];

	const int baseX = blockIdx.x * blockDim.x + threadIdx.x,
		baseY = (blockIdx.y - 1) * blockDim.y + threadIdx.y,
		baseZ = blockIdx.z * blockDim.z + threadIdx.z;

	in += baseZ * (dimx * dimy) + baseY * dimx + baseX;
	out += baseZ * (dimx * (dimy - 2 * size2)) + (baseY - size2) * dimx + baseX;

	if (baseX < dimx && baseY < dimy && baseZ < dimz)
	{
		s_Data[threadIdx.z][threadIdx.x][threadIdx.y + BLOCKDIMY] = (baseY + BLOCKDIMY < dimy) ? in[BLOCKDIMY * dimx] : 0.0f;
		s_Data[threadIdx.z][threadIdx.x][threadIdx.y] = (baseY > 0) ? in[0] : 0.0f;
		s_Data[threadIdx.z][threadIdx.x][threadIdx.y + 2 * BLOCKDIMY] = (dimy - baseY > 2 * BLOCKDIMY) ? in[2 * BLOCKDIMY * dimx] : 0.0f;
		__syncthreads();

		if (baseY + BLOCKDIMY >= size2 && baseY + BLOCKDIMY + size2 < dimy)
		{
			float sum = 0.0f;
#pragma unroll
			for (int i = -size2; i <= size2; i++)
				sum += mask[size2 - i] * s_Data[threadIdx.z][threadIdx.x][threadIdx.y + BLOCKDIMY + i];

			out[BLOCKDIMY * dimx] = sum;
		}
	}
}


__global__ void cudaConvolution_Shared_Memory_3D_Float_Valid_Z_Kernel(float* in, int dimx, int dimy, int dimz, float *mask, int size2, float* out)
{
	__shared__ float s_Data[BLOCKDIMY][BLOCKDIMX][BLOCKDIMZ * 3];

	const int baseX = blockIdx.x * blockDim.x + threadIdx.x,
		baseY = blockIdx.y * blockDim.y + threadIdx.y,
		baseZ = (blockIdx.z - 1) * blockDim.z + threadIdx.z;

	in += baseZ * (dimx * dimy) + baseY * dimx + baseX;
	out += (baseZ - size2) * (dimx * dimy) + baseY * dimx + baseX;

	if (baseX < dimx && baseY < dimy && baseZ < dimz)
	{
		s_Data[threadIdx.y][threadIdx.x][threadIdx.z + BLOCKDIMZ] = (baseZ + BLOCKDIMZ < dimz) ? in[BLOCKDIMZ * dimx * dimy] : 0.0f;
		s_Data[threadIdx.y][threadIdx.x][threadIdx.z] = (baseZ > 0) ? in[0] : 0.0f;
		s_Data[threadIdx.y][threadIdx.x][threadIdx.z + 2 * BLOCKDIMZ] = (dimz - baseZ > 2 * BLOCKDIMZ) ? in[2 * BLOCKDIMZ * dimx * dimy] : 0.0f;
		__syncthreads();

		if (baseZ + BLOCKDIMZ >= size2 && baseZ + BLOCKDIMZ + size2 < dimz)
		{
			float sum = 0.0f;
#pragma unroll
			for (int i = -size2; i <= size2; i++)
				sum += mask[size2 - i] * s_Data[threadIdx.y][threadIdx.x][threadIdx.z + BLOCKDIMZ + i];

			out[BLOCKDIMZ * dimx * dimy] = sum;
		}
	}
}




void cudaConvolution_Shared_Memory_3D_Float_Valid_X(float* d_in, int dimx, int dimy, int dimz, float* d_mask, int maskSize, float* d_out)
{
	dim3 block(10, 10, 10);
	dim3 grid((dimx - 1) / block.x + 1, (dimy - 1) / block.y + 1, (dimz - 1) / block.z + 1);
	cudaConvolution_Shared_Memory_3D_Float_Valid_X_Kernel << <grid, block >> > (d_in, dimx, dimy, dimz, d_mask, maskSize / 2, d_out);
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