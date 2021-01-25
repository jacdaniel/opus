

#include <stdio.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cudaProps.h>


const char* array_cudaDeviceProp[] = {
	"asyncEngineCount",
	"canMapHostMemory",
	"clockRate",
	"computeMode",
	"concurrentKernels",
	"deviceOverlap",
	"ECCEnabled",
	"integrated",
	"kernelExecTimeoutEnabled",
	"l2CacheSize",
	"major",
	"maxGridSize",
	"maxTexture1D",
	"maxTexture1DLayered",
	"maxTexture2D",
	"maxTexture2DLayered",
	"maxTexture3D",
	"maxThreadsDim",
	"maxThreadsPerBlock",
	"maxThreadsPerMultiProcessor",
	"memoryBusWidth",
	"memoryClockRate",
	"memPitch",
	"minor",
	"multiProcessorCount",
	"name",
	"pciBusID",
	"pciDeviceID",
	"pciDomainID",
	"regsPerBlock",
	"sharedMemPerBlock",
	"surfaceAlignment",
	"tccDriver",
	"textureAlignment",
	"totalConstMem",
	"totalGlobalMem",
	"unifiedAddressing",
	"warpSize" };

long cuda_total_memory()
{
	size_t total_memory, free_memory;
	cudaMemGetInfo(&free_memory, &total_memory);
	return (long)total_memory;
}

long cuda_free_memory()
{
	size_t total_memory, free_memory;
	cudaMemGetInfo(&free_memory, &total_memory);
	return (long)free_memory;
}

int cuda_get_nbre_devices()
{
	int count;
	cudaGetDeviceCount(&count);
	return count;
}

int cuda_get_current_device()
{
	int nbre;
	cudaGetDevice(&nbre);
	return nbre;
}

void cuda_set_current_device(int no)
{
	cudaSetDevice(no);
}

long cuda_total_memory2(int no)
{
	size_t total_memory, free_memory;
	int old = cuda_get_current_device();
	cuda_set_current_device(no);
	cudaMemGetInfo(&free_memory, &total_memory);
	cuda_set_current_device(old);
	return (long)total_memory;
}

long cuda_free_memory2(int no)
{
	size_t total_memory, free_memory;
	int old = cuda_get_current_device();
	cuda_set_current_device(no);
	cudaMemGetInfo(&free_memory, &total_memory);
	cuda_set_current_device(old);
	return (long)free_memory;
}







void cuda_props_print(void* _pFile, int id)
{
	int devID = id, i = 0;
	cudaDeviceProp deviceProp;
	cudaError_t error;
	FILE* pFile = (FILE*)_pFile;

	if (devID < 0)
	{
		devID = 0;
		error = cudaGetDevice(&devID);
	}
	error = cudaGetDeviceProperties(&deviceProp, devID);

	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.asyncEngineCount);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.canMapHostMemory);
	fprintf(pFile, "%s: %d Khz\n", array_cudaDeviceProp[i++], deviceProp.clockRate);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.computeMode);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.concurrentKernels);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.deviceOverlap);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.ECCEnabled);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.integrated);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.kernelExecTimeoutEnabled);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.l2CacheSize);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.major);
	fprintf(pFile, "%s: %.0f %d %d\n", array_cudaDeviceProp[i++], (float)deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.maxTexture1D);
	fprintf(pFile, "%s: %d %d\n", array_cudaDeviceProp[i++], deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
	fprintf(pFile, "%s: %d %d\n", array_cudaDeviceProp[i++], deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
	fprintf(pFile, "%s: %d %d %d\n", array_cudaDeviceProp[i++], deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
	fprintf(pFile, "%s: %d %d %d\n", array_cudaDeviceProp[i++], deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
	fprintf(pFile, "%s: %d %d %d\n", array_cudaDeviceProp[i++], deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.maxThreadsPerBlock);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.maxThreadsPerMultiProcessor);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.memoryBusWidth);
	fprintf(pFile, "%s: %d KHz\n", array_cudaDeviceProp[i++], deviceProp.memoryClockRate);
	fprintf(pFile, "%s: %.0f\n", array_cudaDeviceProp[i++], (float)deviceProp.memPitch);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.minor);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.multiProcessorCount);
	fprintf(pFile, "%s: %s\n", array_cudaDeviceProp[i++], deviceProp.name);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.pciBusID);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.pciDeviceID);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.pciDomainID);
	fprintf(pFile, "%s: %d 32bit registers\n", array_cudaDeviceProp[i++], deviceProp.regsPerBlock);
	fprintf(pFile, "%s: %d bits\n", array_cudaDeviceProp[i++], (int)deviceProp.sharedMemPerBlock);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], (int)deviceProp.surfaceAlignment);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], (int)deviceProp.tccDriver);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], (int)deviceProp.textureAlignment);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], (int)deviceProp.totalConstMem);
	fprintf(pFile, "%s: %.0f\n", array_cudaDeviceProp[i++], (float)deviceProp.totalGlobalMem);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.unifiedAddressing);
	fprintf(pFile, "%s: %d\n", array_cudaDeviceProp[i++], deviceProp.warpSize);
}

void cuda_device_capability(int devID, int* capability)
{
	cudaDeviceProp deviceProp;
	cudaError_t error;

	error = cudaGetDevice(&devID);
	error = cudaGetDeviceProperties(&deviceProp, devID);
	capability[0] = deviceProp.major;
	capability[1] = deviceProp.minor;
}

int cuda_device_maxThreadsPerBlock(int devID)
{
	cudaDeviceProp deviceProp;
	cudaError_t error;

	error = cudaGetDevice(&devID);
	error = cudaGetDeviceProperties(&deviceProp, devID);
	return deviceProp.maxThreadsPerBlock;
}

int cuda_device_sharedMemPerBlock(int devID)
{
	cudaDeviceProp deviceProp;
	cudaError_t error;

	error = cudaGetDevice(&devID);
	error = cudaGetDeviceProperties(&deviceProp, devID);
	return (int)deviceProp.sharedMemPerBlock;
}

