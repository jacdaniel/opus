#ifndef __EIGENUTIL__
#define __EIGENUTIL__

#ifndef INV3 
#define INV3  .333333333333333f
#endif

#ifndef INV6
#define INV6  .166666666666667f
#endif

#if !defined (PI_3)
#define PI_3 1.047197551196598f
#endif


#define SPD_3X3_NORMALISE(_sxx, _sxy, _sxz, _syy, _syz, _szz, sxx, sxy, sxz, syy, syz, szz) { \
  float norm; \
	sxx = _sxx; sxy = _sxy; sxz = _sxz; syy = _syy; syz = _syz; szz = _szz; \
	norm = sqrtf(sxx*sxx+syy*syy+szz*szz+2.0f*(sxy*sxy+sxz*sxz+syz*syz)); \
	if ( norm > 0.0f ) \
			{ \
		sxx /= norm; \
		sxy /= norm; \
		sxz /= norm; \
		syy /= norm; \
		syz /= norm; \
		szz /= norm; \
			} \
}


#define EIGEN_SPD_3X3_EIGEN_VECTOR(sxx, sxy, sxz, syy, syz, szz, lambda, ux, uy, uz) { \
	float a = sxx-lambda; \
	float b = syy-lambda; \
	float c = szz-lambda; \
	float t1 = sxy*syz-b*sxz; \
	float t2 = sxz*syz-c*sxy; \
	float t3 = sxz*sxy-a*syz; \
	float vx = t1*t2; \
	float vy = t2*t3; \
	float vz = t1*t3; \
	if ( vx > 0.0f) \
	{ \
		vx = -vx; \
		vy = -vy; \
		vz = -vz;	\
	} \
	float normd = sqrtf(vx*vx+vy*vy+vz*vz);\
	if ( normd == 0.0f ) \
	{ \
		vx = -1.0f; \
		vy = 0.0f; \
		vz = 0.0f; \
	} \
	else \
	{ \
		vx /= normd; \
		vy /= normd; \
		vz /= normd; \
	} \
	ux = vx; \
	uy = vy; \
	uz = vz; \
	}

#define EIGEN_VECTOR_SDP3X3_F(sxx, syy, szz, sxy, sxz, syz, lambda, ux, uy, uz) { \
	float Ai, Bi, Ci, x, y, z, norm; \
	\
	Ai = sxx - lambda; \
	Bi = syy - lambda; \
	Ci = szz - lambda; \
	x = ( sxy * syz - Bi * sxz ) * ( sxz * syz - Ci * sxy ); \
	y = ( sxz * syz - Ci * sxy ) * ( sxz * sxy - Ai * syz ); \
	z = ( sxy * syz - Bi * sxz ) * ( sxz * sxy - Ai * syz ); \
	norm = sqrtf(x*x+y*y+z*z); \
	if ( norm != 0.0f ) \
    { \
		x /= norm; \
		y /= norm; \
		z /= norm; \
	} \
	else \
	{ \
	  x = -1.0f; \
      y = 0.0f; \
		z = 0.0f; \
				}\
	if ( x > 0.0f ) \
			{ \
		x = -x; \
		y = -y; \
		z = -z; \
			} \
	ux = x; \
	uy = y; \
	uz = z; \
}


#define EIGEN_PRINCIPAL_VALUE_SDP3X3_F(sxx, sxy, sxz, syy, syz, szz, lambda) { \
	float I1, I2, I3, s, v, I1_3, I1_3_2, phi, sqrt_v, arg; \
	\
	I1 = sxx + syy + szz; \
	I2 = ( sxx * syy + sxx * szz + syy * szz ) - ( sxy * sxy + sxz * sxz + syz * syz ); \
	I3 = sxx*syy*szz + 2.0f*syz*sxz*syz - ( szz*sxy*sxy + syy*sxz*sxz + sxx*syz*syz ); \
	I1_3 = I1*INV3; \
	I1_3_2 = I1_3*I1_3; \
	v = I1_3_2 - I2*INV3; \
	s = I1_3_2*I1_3 - I1*I2*INV6 + .5f*I3; \
	if ( v <= 0.0f ) \
			{ \
		lambda = 0.0f; \
			} \
				else \
				{ \
		sqrt_v = sqrtf(v); \
		arg = s/(v*sqrt_v); \
		if ( arg > 1.0f ) arg = 1.0f; else if ( arg < -1.0f ) arg = -1.0f; \
		phi = acosf(arg)*INV3; \
		lambda = I1_3 + 2.0f*sqrt_v*cosf(phi); \
				} \
}

// OK
#define EIGEN_SPD_3X3_EIGEN_VALUES(sxx, sxy, sxz, syy, syz, szz, lambda1, lambda2, lambda3){ \
	float i1 = sxx+syy+szz; \
	float i2 = sxx*syy+sxx*szz+syy*szz-(sxy*sxy+sxz*sxz+syz*syz); \
	float i3 = sxx*syy*szz+2.0f*sxy*syz*sxz-(szz*sxy*sxy+syy*sxz*sxz+sxx*syz*syz); \
	float v = fabsf(i1*i1/9.0f-i2/3.0f);	\
	if ( v <= 0.0f ) \
		/*v = 1e-10;*/ \
	{ \
		lambda1 = lambda2 = lambda3 = 0.0f; \
	} \
	else \
	{\
		float s = i1*i1*i1/27.0f-i1*i2/6.0f+i3/2.0f; \
		float d = s/(v*sqrtf(v)); \
		if ( d > 1.0f ) d = 1.0f; else if ( d < -1.0f ) d = -1.0f; \
		float phi = acosf(d)/3.0f; \
		lambda1 = i1/3.0f+2.0f*sqrtf(v)*cosf(phi); \
		lambda2 = i1/3.0f-2.0f*sqrtf(v)*cosf(PI_3+phi); \
		lambda3 = i1 - lambda1 -lambda2; /*i1/3.0-2.0f*sqrt(v)*cos(PI/3.0-phi);*/ \
  }}

// ok
#define EIGEN_PRINCIPAL_VECTOR_SDP3X3_F(_sxx, _syy, _szz, _sxy, _sxz, _syz, ux, uy, uz) { \
	float norm, lambda, sxx = _sxx, sxy = _sxy, sxz = _sxz, syy = _syy, syz = _syz, szz = _szz; \
	SPD_3X3_NORMALISE(_sxx, _sxy, _sxz, _syy, _syz, _szz, sxx, sxy, sxz, syy, syz, szz)  \
	EIGEN_PRINCIPAL_VALUE_SDP3X3_F(sxx, sxy, sxz, syy, syz, szz, lambda) \
	EIGEN_VECTOR_SDP3X3_F(sxx, syy, szz, sxy, sxz, syz, lambda, ux, uy, uz) \
}

// OK
#define SPD_3X3_EIGEN_VALUES_AND_PRINCIPAL_EIGEN_VECTOR(_sxx, _sxy, _sxz, _syy, _syz, _szz, _lambda1, _lambda2, _lambda3, ux, uy, uz){ \
  float sxx_, sxy_, sxz_, syy_, syz_, szz_, lambda1_, lambda2_, lambda3_; \
	SPD_3X3_NORMALISE(_sxx, _sxy, _sxz, _syy, _syz, _szz, sxx_, sxy_, sxz_, syy_, syz_, szz_) \
	EIGEN_SPD_3X3_EIGEN_VALUES(sxx_, sxy_, sxz_, syy_, syz_, szz_, lambda1_, lambda2_, lambda3_) \
  EIGEN_SPD_3X3_EIGEN_VECTOR(sxx_, sxy_, sxz_, syy_, syz_, szz_, lambda1_, ux, uy, uz) \
	_lambda1 = lambda1_; \
	_lambda2 = lambda2_; \
	_lambda3 = lambda3_; \
	}

#endif


