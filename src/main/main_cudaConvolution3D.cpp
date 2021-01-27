
#include <stdio.h>
#include <malloc.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <cudaConvolution_Shared_Memory_Symbol_Mask_3D_Float_Valid_Kernel.h>
#include <cudaConvolution_Shared_Memory_3D_Float_Valid_Kernel.h>
#include <cudaConvolution_3D_Float_Valid_Kernel.h>
#include <gaussian.h>
#include <gaussian.hpp>
#include <xt_file.h>
#include <cudaProps.h>

void dataRead(char* filename, int x1, int x2, int y1, int y2, int z1, int z2, float* out)
{
	XT_FILE* p = new XT_FILE();
	p->openForRead(filename);
	int dimx = p->get_dimx();
	int dimy = p->get_dimy();
	int dimz = p->get_dimz();
	short* data = (short*)calloc(dimx * dimy, sizeof(short));
	int dx = x2 - x1 + 1;
	int dy = y2 - y1 + 1;
	int dz = z2 - z1 + 1;
	for (int z = 0; z < dz; z++)
	{
		p->inlineRead(z + z1, data);
		for (int y = 0; y < dy; y++)
		{
			for (int x = 0; x < dx; x++)
			{
				out[dx * dy * z + dx * y + x] = (float)data[dimx * (y + y1) + x + x1];
			}
		}
	}
	delete p;
}

static void dataWrite(char* refFilename, char *dstFilename, float *data0, int dx, int dy, int dz)
{
	XT_FILE* p = new XT_FILE();
	p->createNew(refFilename, dstFilename, dx, dy, dz, XT_FILE::FLOAT);
	delete p;
	p = new XT_FILE();
	p->openForWrite(dstFilename);
	p->inlineManyWrite(0, dz, data0);
	delete p;
}


int main_cudaConvolution3DKernel(int argc, char** argv)
{
	char* seismicFilename = "D:\\TOTAL\\PLI\\seismic.xt";
	char* resFilename = "D:\\TOTAL\\PLI\\res.xt";

	cuda_props_print(stderr, 0);

	int dimx = 255, dimy = 255, dimz = 255;
	double sigmaG = 1.0;
	int border = 3;
	int dimx0 = dimx + 2 * border, dimy0 = dimy+2*border, dimz0 = dimz+2*border;
	long sizeMask;
	float* in = nullptr, * d_in = nullptr, * out = nullptr, * d_out = nullptr, *d_mask = nullptr, *gx = nullptr, *gy = nullptr, *gz = nullptr;

	float* mask = gaussianMask1D<float>(sigmaG, &sizeMask);
	size_t size = (size_t)dimx * dimy * dimz;
	size_t size0 = (size_t)dimx0 * dimy0 * dimz0;
	mask[0] = mask[1] = mask[2] = mask[4] = mask[5] = mask[6] = 0.0;
	mask[3] = 1.0f;

	in = (float*)calloc(size0, sizeof(float));
	gx = (float*)calloc(size0, sizeof(float));
	gy = (float*)calloc(size0, sizeof(float));
	gz = (float*)calloc(size0, sizeof(float));
	cudaMalloc((void**)&d_mask, sizeMask * sizeof(float));
	cudaMemcpy(d_mask, mask, sizeMask * sizeof(float), cudaMemcpyHostToDevice);

	dataRead(seismicFilename, 0, dimx0 - 1, 0, dimy0 - 1, 0, dimz0 - 1, in);


	cudaMalloc((void**)&d_in, (size_t)size0 * sizeof(float));
	cudaMalloc((void**)&d_out, (size_t)size0 * sizeof(float));

	cudaMemcpy(d_in, in, dimx0 * dimy0 * dimz0 * sizeof(float), cudaMemcpyHostToDevice);
	cudaConvolution_3D_Float_Valid_X(d_in, dimx0, dimy0, dimz0, d_mask, sizeMask, d_out);
	cudaMemcpy(gx, d_out, dimx * dimy0 * dimz0*sizeof(float), cudaMemcpyDeviceToHost);

	cudaConvolution_3D_Float_Valid_Y(d_in, dimx0, dimy0, dimz0, d_mask, sizeMask, d_out);
	cudaMemcpy(gy, d_out, dimx0 * dimy * dimz0*sizeof(float), cudaMemcpyDeviceToHost);

	cudaConvolution_3D_Float_Valid_Z(d_in, dimx0, dimy0, dimz0, d_mask, sizeMask, d_out);
	cudaMemcpy(gz, d_out, dimx0 * dimy0 * dimz*sizeof(float), cudaMemcpyDeviceToHost);


	dataWrite(seismicFilename, resFilename, gx, dimx, dimy0, dimz0);

	FILE* pFile = fopen(resFilename, "wb");
	fwrite(in, sizeof(float), (size_t)dimx0 * dimy0 * dimz0, pFile);
	fclose(pFile);



	// cudaMalloc((void**)&d_in, size0 * sizeof(float));
	// out = (float*)calloc(size0, sizeof(float));
	// cudaMalloc((void**)&d_out, size0 * sizeof(float));

	// cudaMalloc((void**)&d_mask, sizeMask * sizeof(float));
	// cudaMemcpy(d_mask, mask, sizeMask * sizeof(float), cudaMemcpyHostToDevice);

	// for (int i = 0; i < size0; i++) in[i] = (float)(i*i % 100);
	// cudaMemcpy(d_in, in, size0 * sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(in, d_in, size0 * sizeof(float), cudaMemcpyDeviceToHost);

	// cudaConvolution_Shared_Memory_Symbol_Mask_3D_Float_Valid_X(d_in, dimx0, dimy0, dimz0, mask, sizeMask, d_out);
	// cudaConvolution_Shared_Memory_3D_Float_Valid_X(d_in, dimx0, dimy0, dimz0, d_mask, sizeMask, d_out);
	// cudaConvolution_3D_Float_Valid_X(d_in, dimx0, dimy0, dimz0, d_mask, sizeMask, d_out);

	// cudaMemcpy(out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);


	return 0;
}