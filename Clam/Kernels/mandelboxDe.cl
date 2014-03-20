#ifndef MaxIters
#define MaxIters 64
#endif

#ifndef Bailout
#define Bailout 128
#endif

#ifndef Scale
#define Scale -1.5
#endif

#ifndef FoldingLimit
#define FoldingLimit 1.0
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
	float dz = 1.0;
	for (int n = 0; n < MaxIters; n++) {
		z = clamp(z, -FoldingLimit, FoldingLimit) * 2.0 - z;

		float r2 = dot(z, z);

		if (r2 > Bailout)
			break;

		if (r2 < MinRadius2) { 
			float temp = FixedRadius2 / MinRadius2;
			z *= temp;
			dz *= temp;
		} else if (r2 < FixedRadius2) { 
			float temp = FixedRadius2 / r2;
			z *= temp;
			dz *= temp;
		}

		z = Scale * z + offset;
		dz = dz * fabs(Scale) + 1.0;
	}
	return length(z) / dz;
}
