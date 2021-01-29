
#include <cuda_runtime.h>

#include "cudaNormalToDip.h"

static __global__ void normalFloatToDipInt16Kernel(float* nx, float* ny, float* nz, long size, short* dipxy, short* dipxz)
{
	const int add = blockIdx.x * blockDim.x + threadIdx.x;
	if (add < size)
	{
		if (nx[add] != 0.0f)
		{
			float dip = -ny[add] / nx[add];
			dipxy[add] = (short)(1000.0f * dip);
			dip = -nz[add] / nx[add];
			dipxz[add] = (short)(1000.0f * dip);
		}
		else
		{
			dipxy[add] = 0;
			dipxz[add] = 0;
		}
	}
}

NormalToDip::NormalToDip()
{
	setCudaNormal(nullptr, nullptr, nullptr);
	setDip(nullptr, nullptr);
	setSize(1, 1, 1);
	setNormalFormat(NormalToDip::FLOAT);
	setDipFormat(NormalToDip::INT16);
}

NormalToDip::~NormalToDip()
{
}

void NormalToDip::setCudaNormal(void* d_nx, void* d_ny, void* d_nz)
{
	this->d_nx = d_nx;
	this->d_ny = d_ny;
	this->d_nz = d_nz;
}

void NormalToDip::setDip(void* d_dipxy, void* d_dipxz)
{
	this->d_dipxy = d_dipxy;
	this->d_dipxz = d_dipxz;
}

void NormalToDip::setSize(int dimx, int dimy, int dimz)
{
	size[0] = dimx;
	size[1] = dimy;
	size[2] = dimz;
}

void NormalToDip::setNormalFormat(int format)
{
	normalFormat = format;
}

void NormalToDip::setDipFormat(int format)
{
	dipFormat = format;
}

void NormalToDip::run()
{
	size_t size0 = (size_t)size[0] * size[1] * size[2];
	dim3 block(1024);
	dim3 grid((size0 - 1) / block.x + 1);
	if (dipFormat == NormalToDip::INT16)
	{		
		normalFloatToDipInt16Kernel << <grid, block >> > ((float*)d_nx, (float*)d_ny, (float*)d_nz, size0, (short*)d_dipxy, (short*)d_dipxz);
		cudaThreadSynchronize();
	}
}
