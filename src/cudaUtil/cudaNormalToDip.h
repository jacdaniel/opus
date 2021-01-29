
#ifndef __CUDANORMALTODIP__
#define __CUDANORMALTODIP__

class NormalToDip
{
public:
	enum normalFormat { INT16, FLOAT, DOUBLE };
private:
	int size[3];
	void* d_nx, * d_ny, * d_nz;
	void* d_dipxy, * d_dipxz;
	int normalFormat, dipFormat;

public:
	NormalToDip();
	~NormalToDip();
	void setCudaNormal(void* d_nx, void* d_ny, void* d_nz);
	void setDip(void* d_dipxy, void* d_dipxz);
	void setSize(int dimx, int dimy, int dimz);
	void setNormalFormat(int format);
	void setDipFormat(int format);
	void run();



};


#endif