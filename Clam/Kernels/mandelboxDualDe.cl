#ifndef MaxIters
#define MaxIters 64
#endif

#ifndef Bailout
#define Bailout 128
#endif

#ifndef ScaleX
#define ScaleX -1.5
#endif

#ifndef ScaleY
#define ScaleY -2.5
#endif

#ifndef ScaleZ
#define ScaleZ -3.0
#endif

#ifndef FoldingLimitX
#define FoldingLimitX 1.0
#endif

#ifndef FoldingLimitY
#define FoldingLimitY 1.0
#endif

#ifndef FoldingLimitZ
#define FoldingLimitZ 1.0
#endif

#ifndef FixedRadius2
#define FixedRadius2 1.0
#endif

#ifndef MinRadius2
#define MinRadius2 0.125
#endif

float De(float3 z)
{
#ifndef JuliaCenter
	float3 offset = z;
#else
	float3 offset = (float3)(JuliaCenter);
#endif
	float dzX = 1.0;
	float dzY = 1.0;
	float dzZ = 1.0;
	for (int n = 0; n < MaxIters; n++) {
		dzX *= 1.0 - 2.0 * step(FoldingLimitX, fabs(z.x));
		dzY *= 1.0 - 2.0 * step(FoldingLimitY, fabs(z.y));
		dzZ *= 1.0 - 2.0 * step(FoldingLimitZ, fabs(z.z));
		z = clamp(z, -(float3)(FoldingLimitX,FoldingLimitY,FoldingLimitZ), (float3)(FoldingLimitX,FoldingLimitY,FoldingLimitZ)) * 2.0 - z;

		float r2 = dot(z, z);

		if (r2 > Bailout)
			break;

		if (r2 < MinRadius2) { 
			float temp = FixedRadius2 / MinRadius2;
			z *= temp;
			dzX *= temp;
			dzY *= temp;
			dzZ *= temp;
		} else if (r2 < FixedRadius2) { 
			float temp = FixedRadius2 / r2;
			z *= temp;
			dzX *= temp;
			dzY *= temp;
			dzZ *= temp;
		}

		z = (float3)(ScaleX, ScaleY, ScaleZ) * z + offset;
		dzX = fabs(ScaleX) * dzX + 1.0;
		dzY = fabs(ScaleY) * dzY + 1.0;
		dzZ = fabs(ScaleZ) * dzZ + 1.0;
	}
	float dotzz = dot(z, z);
	return dotzz / sqrt(dotzz * dzX * dzX + dotzz * dzY * dzY + dotzz * dzZ * dzZ);
}

float3 DeColor(float3 z)
{
	return (float3)(1); // TODO
}
