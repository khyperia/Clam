#ifndef MaxIters
#define MaxIters 2048
#endif

#ifndef Bailout
#define Bailout 2
#endif

float3 GetColor(flt i)
{
	return (float3)(sin(i / 17.0) * 0.5 + 0.5, sin(i / 19.0) * 0.5 + 0.5, sin(i / 23.0) * 0.5 + 0.5);
}

flt ComputeSmooth(flt2 last)
{
	return 1 + log2(log((flt)Bailout) / log(length(last)));
}

float3 IterateAlt(flt2 z, flt2 initZ)
{
	flt2 c = z;
	z += initZ;
	for (int it = 0; it < MaxIters; it++)
    {
		flt x2 = z.x * z.x;
		flt y2 = z.y * z.y;
		if (x2 + y2 > Bailout * Bailout)
			return GetColor(ComputeSmooth(z) + it);
		z = (flt2)(x2 - y2, 2 * z.x * z.y) + c;
    }
	return (float3)(0,0,0);
}

float3 Iterate(flt2 z)
{
	return IterateAlt(z, (flt2)(0,0));
}
