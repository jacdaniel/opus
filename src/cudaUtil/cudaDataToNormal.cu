
#include <cudaDataToNormal.h>

cudaDataToNormal::PARAM::PARAM()
{
}

cudaDataToNormal::PARAM::~PARAM()
{
}


cudaDataToNormal::cudaDataToNormal()
{
	setDataIn(nullptr);
	setNormal(nullptr, nullptr, nullptr);
	setLambda(nullptr, nullptr, nullptr);
	sigmaGradient(1.0);
	sigmaTensor(1.5);
	setSize(1, 1, 1);
}

cudaDataToNormal::~cudaDataToNormal()
{

}

void cudaDataToNormal::setDataIn(void* in)
{
	this->in = in;
}

void cudaDataToNormal::setNormal(void* nx, void* ny, void* nz)
{
	this->nx = nx;
	this->ny = ny;
	this->nz = nz;
}

void cudaDataToNormal::setLambda(void* lambda1, void* lambda2, void* lambda3)
{
	this->lambda1 = lambda1;
	this->lambda2 = lambda2;
	this->lambda3 = lambda3;
}

void cudaDataToNormal::sigmaGradient(double sigma)
{
	this->sigmaG = sigma;
}

void cudaDataToNormal::sigmaTensor(double sigma)
{
	this->sigmaT = sigma;
}

void cudaDataToNormal::setSize(int dimx, int dimy, int dimz)
{
	size[0] = dimx;
	size[1] = dimy;
	size[2] = dimz;
}
