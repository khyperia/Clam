int Iterate(float2 z)
{
	float x = z.x;
	float y = z.y;
	float xc = x;
	float yc = y;
	for (int it = 0; it < 256 * 8; it++)
    {
		float x2 = x * x;
		float y2 = y * y;
		if (x2 + y2 > 4)
			return it;
		float twoxy = 2 * x * y;
		x = x2 - y2 + xc;
		y = twoxy + yc;
    }
	return 0;
}

int IterateAlt(float2 z, float2 c)
{
	float x = z.x;
	float y = z.y;
	float xc = c.x;
	float yc = c.y;
	for (int it = 0; it < 256 * 8; it++)
    {
		float x2 = x * x;
		float y2 = y * y;
		if (x2 + y2 > 4)
			return it;
		float twoxy = 2 * x * y;
		x = x2 - y2 + xc;
		y = twoxy + yc;
    }
	return 0;
}
