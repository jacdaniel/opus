#ifndef __CUDATENSORTOEIGENVALUEANDPRINCIPALEIGENVECTOR_3D_FLOAT__
#define __CUDATENSORTOEIGENVALUEANDPRINCIPALEIGENVECTOR_3D_FLOAT__


void cudaTensorToEigenValueAndPrincipalEigenVector_3D_Float(float* d_Txx, float* d_Txy, float* d_Txz, float* d_Tyy, float* d_Tyz, float* d_Tzz,
	long size, float* d_lambda1, float* d_lambda2, float* d_lambda3, float* d_nx, float* d_ny, float* d_nz);


#endif
