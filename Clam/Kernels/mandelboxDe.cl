#ifndef Scale
#define Scale -1.5f
#endif

#define FoldingLimit 1.0f
#define FixedRadius2 1.0f
#define MinRadius2 0.125f

float De(float3 z)
{
	float3 offset = z;
	float dz = 1.0f;
	for (int n = 0; n < 2048; n++) {
		z = clamp(z, -FoldingLimit, FoldingLimit) * 2.0f - z;

		float r2 = dot(z, z);

		if (r2 > 65536)
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
		dz = dz * fabs(Scale) + 1.0f;
	}
	return length(z) / dz;
}
