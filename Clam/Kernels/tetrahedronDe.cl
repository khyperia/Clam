#ifndef Iterations
#define Iterations 16
#endif

#ifndef Offset
#define Offset 1.0
#endif

#ifndef Scale
#define Scale 2.0
#endif

float De(float3 z)
{
	float r;
	int n;
	for (n = 0; n < Iterations; n++)
	{
		if(z.x + z.y < 0)
			z.xy = -z.yx;
		if(z.x + z.z < 0)
			z.xz = -z.zx;
		if(z.y + z.z < 0)
			z.zy = -z.yz;
		z = z * Scale - Offset * (Scale - 1.0);
	}
	return length(z) * pow(Scale, (float)-n);
}

float3 DeColor(float3 z)
{
	return (float3)(1); // TODO
}
