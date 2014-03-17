#ifndef SupersampleSize
#define SupersampleSize 1
#endif

__kernel void Main(__global float4* screen, int screenWidth, int width, int height, flt xCenter, flt yCenter, flt zoom)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x >= width || y >= height)
		return;

	float3 accum = (float3)(0, 0, 0);
	for (int dx = 0; dx < SupersampleSize; dx++)
	{
		for (int dy = 0; dy < SupersampleSize; dy++)
		{
			flt2 screenCoords = (flt2)((float)(x * SupersampleSize + dx) / (width * SupersampleSize) * 2 - 1, ((float)(y * SupersampleSize + dy) / (height * SupersampleSize) * 2 - 1) * height / width);
			flt2 coordinates = screenCoords * zoom + (flt2)(xCenter, yCenter);
			accum += Iterate(coordinates);
		}
	}
	int screenIndex = screenWidth ? (y - get_global_offset(1)) * screenWidth + (x - get_global_offset(0)) : y * width + x;
	screen[screenIndex] = (float4)(accum / (SupersampleSize * SupersampleSize), 1);
}
