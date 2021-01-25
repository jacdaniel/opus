
#include <math.h>
#include <stdio.h>
#include <malloc.h>

#include <gaussian.h>


template<typename T> T* gaussianMask1D(double sigma, long* size)
{
	T* mask = nullptr;
	double norm = 0.0, x, den;
	if (sigma == 0.0) return nullptr;
	long width = 2 * (long)ceil(3.0 * sigma) + 1 /*gaussSigma2Size(sigma)*/, i, width_2;
	width_2 = width / 2;
	den = 2.0 * sigma * sigma;
	mask = (T*)calloc(width, sizeof(T));
	for (i = 0; i < width; i++)
	{
		x = (double)(i - width_2);
		mask[i] = (T) exp(-(x * x) / den);
		norm += (double)mask[i];
	}
	for (i = 0; i < width; i++)
		mask[i] /= (float)norm;
	if (size)
		*size = width;
	return mask;
}

template<typename T> T* gaussianGradMask1D(double sigma, long* size)
{
	T* mask = nullptr;
	double  norm = 0.0, x, den;
	if (sigma == 0.0) { *size = 0; return NULL; }

	long width = 2 * (long)ceil(3.0 * sigma) + 1 /*gaussSigma2Size(sigma)*/, i, width_2;

	width_2 = width / 2;

	den = 2.0 * sigma * sigma;
	mask = (T*)calloc(width, sizeof(T));
	for (i = 0; i < width; i++)
	{
		x = (double)(i - width_2);
		mask[i] = (T)(-x * exp(-(x * x) / den));
		// norm += fabs(mask[i]);	
		norm += (double) mask[i] * (width / 2 - i);
	}
	for (i = 0; i < width; i++)
		mask[i] /= (T)norm;
	if (size)
		*size = width;
	return mask;
}


template<typename T> T* gaussianMaskFree(T* mask)
{
	if (mask == nullptr) return nullptr;
	free(mask);
	return nullptr;
}
