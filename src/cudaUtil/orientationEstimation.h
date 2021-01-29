
#ifndef __ORIENTATIONESTIMATION__
#define __ORIENTATIONESTIMATION__

#include "cudaNormalToDip.h"


typedef struct _OrientationEstimationParam
{
	void* d_in, * d_out;
	void* d_gx, * d_gy, * d_gz, ** d_Tii;
	// *d_Txx, * d_Txy, * d_Txz, * d_Tyy, * d_Tyz, * d_Tzz;
	void* d_nx, * d_ny, * d_nz;
	void* d_temp1, * d_temp2;
	float* d_LPGrad, * d_HPGrad, * d_LPTens, *lpGrad, *hpGrad, *lpTens;
	long lpGradSize, hpGradSize, lpTensSize;
	int gradSize[3], tensSize[3];
	size_t size0, gradSize0, tensSize0;
	NormalToDip* normalToDip;
}OrientationEstimationParam;

class OrientationEstimation
{
public:
	enum TensorIdx {ID_TXX, ID_TXY, ID_TXZ, ID_TYY, ID_TYZ, ID_TZZ};
private:
	void* dataIn, * dataOut;
	int dataInSize[3];
	double sigmaGradient, sigmaTensor;
	int cpuGpu;
	void* d_dipxy, * d_dipxz;
	
	OrientationEstimationParam* param;

public:
	OrientationEstimation();
	~OrientationEstimation();
	void setDataIn(void* data);
	void setDataInSize(int dimx, int dimy, int dimz);
	void setDipOut(void* d_dipxy, void* d_dipxz);
	void setDataOut(void* data);
	void setSigmaGradient(double sigma);
	void setSigmaTensor(double sigma);
	void setCpuGpu(int val);
	void run();

	int paramInit(OrientationEstimationParam **param);
	int paramRelease(OrientationEstimationParam** param);



};




#endif