#ifndef __UTIL0__
#define __UTIL0__

#define SUCCESS 0
#define FAIL 1


int callocSafe(void** ptr, size_t size0, size_t sizeof_, char *file, int line);
#define CALLOCSAFE(ptr, size0, type) callocSafe((void**)ptr, size0, sizeof(type), __FILE__, __LINE__);

int cudaMallocSafe(void** ptr, size_t size, char *file, int line);
#define CUDAMALLOCSAFE(ptr, size) cudaMallocSafe((void**)ptr, size, __FILE__, __LINE__);

#define FREESAFE(ptr) { if ( (ptr) != nullptr ) { free(ptr); (ptr) = nullptr; } }

int cudaFreeSafe(void** ptr, char* file, int line);
#define CUDAFREESAFE(ptr) cudaFreeSafe((void**)ptr, __FILE__, __LINE__);

#define DELETESAFE(ptr) { if ( ptr != nullptr ) { delete ptr; ptr = nullptr; }}


//

template <typename T> void cudaWriteFile(char* filename, T* d_data, size_t size)
{
	FILE* pFile;
	T* data = nullptr;
	CALLOCSAFE(&data, size, sizeof(T));
	cudaMemcpy(data, d_data, size * sizeof(T), cudaMemcpyDeviceToHost);
	pFile = fopen(filename, "wb");
	fwrite(data, sizeof(T), size, pFile);
	fclose(pFile);
	FREESAFE(data)
}

#endif
