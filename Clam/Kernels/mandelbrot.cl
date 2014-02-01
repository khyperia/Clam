#ifndef MaxIters
#define MaxIters 2048
#endif

int Iterate(float2 z)
{
	float2 c = z;
	for (int it = 0; it < MaxIters; it++)
    {
		float x2 = z.x * z.x;
		float y2 = z.y * z.y;
		if (x2 + y2 > 4)
			return it;
		z = (float2)(x2 - y2, 2 * z.x * z.y) + c;
    }
	return 0;
}

int IterateAlt(float2 z, float2 c)
{
	for (int it = 0; it < MaxIters; it++)
    {
		float x2 = z.x * z.x;
		float y2 = z.y * z.y;
		if (x2 + y2 > 4)
			return it;
		z = (float2)(x2 - y2, 2 * z.x * z.y) + c;
    }
	return 0;
}
