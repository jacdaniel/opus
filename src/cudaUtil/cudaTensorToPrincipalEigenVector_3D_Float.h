#ifndef __CUDATENSORTOPRINCIPALEIGANVECTOR__
#define __CUDATENSORTOPRINCIPALEIGANVECTOR__

void cudaTensorToPrincipalEigenVector_3D_Float(float* d_Txx, float* d_Txy, float* d_Txz, float* d_Tyy, float* d_Tyz, float* d_Tzz,
	long size, float* d_nx, float *d_ny, float* d_nz);

#endif