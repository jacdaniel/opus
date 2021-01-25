#ifndef __CUDAPROPS__
#define __CUDAPROPS__



long cuda_total_memory();

long cuda_free_memory();

int cuda_get_nbre_devices();

int cuda_get_current_device();

void cuda_set_current_device(int no);

long cuda_total_memory2(int no);

long cuda_free_memory2(int no);


void cuda_props_print(void* _pFile, int id);


void cuda_device_capability(int devID, int* capability);


int cuda_device_maxThreadsPerBlock(int devID);


int cuda_device_sharedMemPerBlock(int devID);






#endif
