
#include <stdio.h>
#include <malloc.h>

#include <cuda_runtime.h>

#include <util0.h>


int callocSafe(void** ptr, size_t size0, size_t sizeof_, char* file, int line)
{
	if (ptr == nullptr ) { fprintf(stderr, "callocSafe error - null pointer: %s %d\n", file, line); return FAIL; }
	*ptr = calloc(size0, sizeof_);
	if (*ptr == nullptr)
	{
		fprintf(stderr, "callocSafe error: %s %d\n", file, line);
		fprintf(stderr, "cannot allocate %zd bytes of sizeof %d\n", size0, sizeof_);
		return FAIL;
	}
	return SUCCESS;
}


int cudaMallocSafe(void** ptr, size_t size, char* file, int line)
{
	if ( ptr == nullptr ) { fprintf(stderr, "cudaMallocSafe error - null pointer: %s %d\n", file, line); return FAIL; }
	cudaError ret = cudaMalloc(ptr, size);
	if (ret != cudaSuccess)
	{
		fprintf(stderr, "cudaMallocSafe error [ %d ]: %s %d\n", ret, file, line);
		fprintf(stderr, "cannot allocate %zd bytes\n", size);
		return FAIL;
	}
	return SUCCESS;
}

int cudaFreeSafe(void** ptr, char* file, int line)
{
	if (ptr == nullptr || *ptr == nullptr ) { return SUCCESS; }
	cudaError ret = cudaFree(*ptr);
	if (ret != cudaSuccess)
	{
		fprintf(stderr, "cudaFreeSafe error [ %d ]: %s %d\n", ret, file, line);
		return FAIL;
	}
	return SUCCESS;
}

