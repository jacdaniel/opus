
class cudaDataToNormal
{
	enum TensorIndex {TXX, TXY, TXZ, TYY, TYZ, TZZ};
private:

	class PARAM
	{
	public:
		PARAM();
		~PARAM();
	};

	void* in, * nx, * ny, * nz, *lambda1, *lambda2, *lambda3;
	void* d_in, * d_nx, * d_ny, * d_nz, * d_lambda1, * d_lambda2, * d_lambda3;
	void* d_gx, * d_gy, * d_gz;
	double sigmaG, sigmaT;
	void* Txx[6];
	int size[3];
	PARAM *param;

public:
	cudaDataToNormal();
	~cudaDataToNormal();
	void setDataIn(void* in);
	void setNormal(void* nx, void* ny, void* nz);
	void setLambda(void* lambda1, void* lambda2, void* lambda3);
	void sigmaGradient(double sigma);
	void sigmaTensor(double sigma);
	void setSize(int dimx, int dimy, int dimz);
};