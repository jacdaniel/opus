
#include <math.h>
#include <gaussian.h>


long gaussSigma2Size(double sigma)
{
	return 2 * (long)ceil(3.0 * sigma) + 1;
}


