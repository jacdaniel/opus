
#include <stdio.h>
#include <malloc.h>

#include <cuda_runtime.h>
#include <util0.h>
#include <gaussian.hpp>

#include <cudaGradient_3D_Float_Valid_Kernel.h>
#include <cudaGradientToTensor_3D_Float_Valid_Kernel.h>
#include <cudaTensorToEigenValueAndPrincipalEigenVector_3D_Float.h>
#include <cudaTensorToPrincipalEigenVector_3D_Float.h>

#include <orientationEstimation.h>

OrientationEstimation::OrientationEstimation()
{
	setDataIn(nullptr);

	param = nullptr;
}

OrientationEstimation::~OrientationEstimation()
{

}

void OrientationEstimation::setDataIn(void* data)
{
	dataIn = data;
}

void OrientationEstimation::setDataInSize(int dimx, int dimy, int dimz)
{
	dataInSize[0] = dimx;
	dataInSize[1] = dimy;
	dataInSize[2] = dimz;
}

void OrientationEstimation::setDataOut(void* data)
{
	dataOut = data;
}

void OrientationEstimation::setDipOut(void* d_dipxy, void* d_dipxz)
{
	this->d_dipxy = d_dipxy;
	this->d_dipxz = d_dipxz;
}

void OrientationEstimation::setSigmaGradient(double sigma)
{
	sigmaGradient = sigma;
}

void OrientationEstimation::setSigmaTensor(double sigma)
{
	sigmaTensor = sigma;
}

void OrientationEstimation::setCpuGpu(int val)
{
	cpuGpu = val;
}

int OrientationEstimation::paramInit(OrientationEstimationParam **param)
{
	int ret = SUCCESS, ret0;

	CALLOCSAFE(param, 1, OrientationEstimationParam)

	(*param)->lpGrad = (float*)gaussianMask1D<float>(sigmaGradient, &(*param)->lpGradSize);
	(*param)->hpGrad = (float*)gaussianGradMask1D<float>(sigmaGradient, &(*param)->hpGradSize);
	(*param)->lpTens = (float*)gaussianMask1D<float>(sigmaTensor, &(*param)->lpTensSize);
	
	CUDAMALLOCSAFE(&(*param)->d_LPGrad, (*param)->lpGradSize * sizeof(float));
	cudaMemcpy((*param)->d_LPGrad, (*param)->lpGrad, (*param)->lpGradSize * sizeof(float), cudaMemcpyHostToDevice);
	CUDAMALLOCSAFE(&(*param)->d_HPGrad, (*param)->hpGradSize * sizeof(float));
	cudaMemcpy((*param)->d_HPGrad, (*param)->hpGrad, (*param)->hpGradSize * sizeof(float), cudaMemcpyHostToDevice);
	CUDAMALLOCSAFE(&(*param)->d_LPTens, (*param)->lpTensSize * sizeof(float));
	cudaMemcpy((*param)->d_LPTens, (*param)->lpTens, (*param)->lpTensSize * sizeof(float), cudaMemcpyHostToDevice);

	for (int n = 0; n < 3; n++)
	{
		if (dataInSize[n] > (*param)->lpGradSize )
		{
			(*param)->gradSize[n] = dataInSize[n] - (*param)->lpGradSize + 1;
		}

		if ((*param)->gradSize[n] > (*param)->lpTensSize)
		{
			(*param)->tensSize[n] = (*param)->gradSize[n] - (*param)->lpTensSize + 1;
		}
	}

	(*param)->size0 = (size_t)dataInSize[0] * dataInSize[1] * dataInSize[2];
	(*param)->gradSize0 = (size_t)(*param)->gradSize[0] * (size_t)(*param)->gradSize[1] * (size_t)(*param)->gradSize[2];
	(*param)->tensSize0 = (size_t)(*param)->tensSize[0] * (size_t)(*param)->tensSize[1] * (size_t)(*param)->tensSize[2];

	ret = CALLOCSAFE(&(*param)->d_gi, 3, float*);
	for (int i = 0; i < 3; i++)
	{
		ret0 = CUDAMALLOCSAFE(&(*param)->d_gi[i], (*param)->gradSize0 * sizeof(float));
	}

	ret = CALLOCSAFE(&(*param)->d_Tii, 6, float*);
	for (int i = 0; i < 6; i++)
	{
		ret0 = CUDAMALLOCSAFE(&(*param)->d_Tii[i], (*param)->tensSize0 * sizeof(float));
	}

	ret0 = CUDAMALLOCSAFE(&(*param)->d_nx, (*param)->tensSize0 * sizeof(float));
	ret0 = CUDAMALLOCSAFE(&(*param)->d_ny, (*param)->tensSize0 * sizeof(float));
	ret0 = CUDAMALLOCSAFE(&(*param)->d_nz, (*param)->tensSize0 * sizeof(float));

	ret0 = CUDAMALLOCSAFE(&(*param)->d_temp1, (*param)->size0 * sizeof(float));
	ret0 = CUDAMALLOCSAFE(&(*param)->d_temp2, (*param)->size0 * sizeof(float));

	(*param)->normalToDip = new NormalToDip();
	(*param)->normalToDip->setCudaNormal((*param)->d_nx, (*param)->d_ny, (*param)->d_nz);
	(*param)->normalToDip->setSize((*param)->tensSize[0], (*param)->tensSize[1], (*param)->tensSize[1]);
	(*param)->normalToDip->setDip(d_dipxy, d_dipxz);
	(*param)->normalToDip->setNormalFormat(NormalToDip::FLOAT);
	(*param)->normalToDip->setDipFormat(NormalToDip::INT16);
	
	return ret;
}

int OrientationEstimation::paramRelease(OrientationEstimationParam** param)
{
	if (param == nullptr || *param == nullptr) return SUCCESS;
	for (int i = 0; i < 3; i++)
	{
		CUDAFREESAFE(&(*param)->d_gi[i])
	}
	FREESAFE((*param)->d_gi)

	for (int i = 0; i < 6; i++)
	{
		CUDAFREESAFE(&(*param)->d_Tii[i])
	}
	FREESAFE((*param)->d_Tii)
	
	CUDAFREESAFE(&(*param)->d_nx)
	CUDAFREESAFE(&(*param)->d_ny)
	CUDAFREESAFE(&(*param)->d_nz)
	FREESAFE(*param);
	return SUCCESS;
}



void OrientationEstimation::run()
{
	if (param == nullptr)
	{
		int ret = paramInit(&param);
		if (ret != SUCCESS)
		{
			paramRelease(&param);
			return;
		}
	}

	cudaGradient_3D_Float_Valid((float*)dataIn, dataInSize[0], dataInSize[1], dataInSize[2], 
		param->d_LPGrad, param->lpGradSize, 
		param->d_HPGrad, param->hpGradSize, 
		(float*)param->d_temp1, (float*)param->d_temp2, 
		(float*)param->d_gi[ID_GX], (float*)param->d_gi[ID_GY], (float*)param->d_gi[ID_GZ]);

	cudaGradientToTensor_3D_Float_Valid_Kernel((float*)param->d_gi[ID_GX], (float*)param->d_gi[ID_GY], (float*)param->d_gi[ID_GZ], param->gradSize[0], param->gradSize[1], param->gradSize[2],
		param->d_LPTens, param->lpTensSize,
		(float*)param->d_temp1, (float*)param->d_temp2,
		(float*)param->d_Tii[ID_TXX], (float*)param->d_Tii[ID_TXY], (float*)param->d_Tii[ID_TXZ],
		(float*)param->d_Tii[ID_TYY], (float*)param->d_Tii[ID_TYZ],
		(float*)param->d_Tii[ID_TZZ]);

	cudaTensorToPrincipalEigenVector_3D_Float((float*)param->d_Tii[ID_TXX], (float*)param->d_Tii[ID_TXY], (float*)param->d_Tii[ID_TXZ], (float*)param->d_Tii[ID_TYY], 
		(float*)param->d_Tii[ID_TYZ], (float*)param->d_Tii[ID_TZZ], param->tensSize0, 
		(float*)param->d_nx, (float*)param->d_ny, (float*)param->d_nz);

	param->normalToDip->run();
}

