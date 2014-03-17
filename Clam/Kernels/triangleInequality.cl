#ifndef MaxIters
#define MaxIters 2048
#endif

#ifndef Bailout
#define Bailout 1024
#endif

flt2 Square(flt2 z)
{
	return (flt2)(z.x * z.x - z.y * z.y, 2 * z.x * z.y);
}

float3 GetColor(flt i)
{
	i = log(i) * 100;
	return (float3)(sin(i / 17.0) * 0.5 + 0.5, sin(i / 19.0) * 0.5 + 0.5, sin(i / 23.0) * 0.5 + 0.5);
}

flt GetT(flt2 z, flt2 newZ, flt2 c)
{
	flt zLenSq = length(Square(z));
	flt cLen = length(c);
	flt m = fabs(zLenSq - cLen);
	flt M = zLenSq + cLen;
	flt t = (length(newZ) - m) / (M - m);
	return t;
}

flt ComputeSmooth(flt2 last)
{
	return 1 + log2(log(sqrt((flt)Bailout)) / log(length(last)));
}

float3 IterateAlt(flt2 z, flt2 c)
{
	flt2 newZ;
	flt sumPrev = 0.0;
	flt sum = 0.0;
	for (int i = 0; i < MaxIters; i++)
	{
		newZ = Square(z) + c;
		if (dot(newZ, newZ) > Bailout)
		{
			flt smooth = ComputeSmooth(newZ);
			flt color = sum / (i - 2) * smooth + sumPrev / (i - 3) * (1 - smooth);
			return GetColor(color);
		}
		sumPrev = sum;
		sum += GetT(z, newZ, c);
		z = newZ;
	}
	return (float3)(0, 0, 0);
}

float3 Iterate(flt2 coordinates)
{
	return IterateAlt(coordinates, coordinates);
}
