
#include <stdio.h>
#include <malloc.h>

#include <cuda_runtime.h>
#include <util0.h>
#include <orientationEstimation.h>

void dataRead(char* filename, int x1, int x2, int y1, int y2, int z1, int z2, float* out);
void syntheticDataCreate_v1(int dimx, int dimy, int dimz, float* data);

int main_orientationEstimation(int argc, char** argv)
{
	char* seismicFilename = "D:\\TOTAL\\PLI\\seismic.xt";

	int dimx = 101, dimy = 101, dimz = 101;
	float* d_in = nullptr, *in = nullptr;
	short* d_dipxy = nullptr, * d_dipxz = nullptr, *dipxy = nullptr, *dipxz = nullptr;

	size_t sizeIn = (size_t)dimx * dimy * dimz;
	size_t sizeOut = (size_t)(dimx - 12) * (dimy - 12) * (dimz - 12);
	CALLOCSAFE(&in, sizeIn, sizeof(float));
	CUDAMALLOCSAFE(&d_in, sizeIn * sizeof(float));
	CALLOCSAFE(&dipxy, sizeOut, sizeof(float));
	CALLOCSAFE(&dipxz, sizeOut, sizeof(float));
	CUDAMALLOCSAFE(&d_dipxy, sizeOut * sizeof(float));
	CUDAMALLOCSAFE(&d_dipxz, sizeOut * sizeof(float));

	dataRead(seismicFilename, 0, dimx - 1, 0, dimy - 1, 0, dimz - 1, in);
	syntheticDataCreate_v1(dimx, dimy, dimz, in);
	cudaMemcpy(d_in, in, sizeIn * sizeof(float), cudaMemcpyHostToDevice);




	OrientationEstimation* p = new OrientationEstimation();
	p->setDataInSize(dimx, dimy, dimz);
	p->setCpuGpu(1);
	p->setSigmaGradient(1.0);
	p->setSigmaTensor(1.0);
	p->setDataIn(d_in);
	p->setDipOut(d_dipxy, d_dipxz);
	p->run();

	cudaMemcpy(dipxy, d_dipxy, sizeOut * sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpy(dipxz, d_dipxz, sizeOut * sizeof(short), cudaMemcpyDeviceToHost);

	delete p;

	FILE *pFile = fopen("d:\\dipxy.raw", "wb");
	fwrite(dipxy, sizeof(short), sizeOut, pFile);
	fclose(pFile);

	pFile = fopen("d:\\dipxz.raw", "wb");
	fwrite(dipxz, sizeof(short), sizeOut, pFile);
	fclose(pFile);
	return 0;
}