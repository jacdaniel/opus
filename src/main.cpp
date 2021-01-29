#include <stdio.h>

int main_cudaConvolution3DKernel(int argc, char** argv);
int main_cudaGradient(int argc, char** argv);
int main_orientationEstimation(int argc, char** argv);


int main(int argc, char** argv)
{

	// main_cudaConvolution3DKernel(argc, argv);
	// main_cudaGradient(argc, argv);
	main_orientationEstimation(argc, argv);
	fprintf(stderr, "ok\n");
}