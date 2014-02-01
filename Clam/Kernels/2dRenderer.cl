#ifndef SupersampleSize
#define SupersampleSize 1
#endif

__kernel void Main(__global float4* screen, int width, int height, float xCenter, float yCenter, float zoom)
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
			float2 screenCoords = (float2)((float)(x * SupersampleSize + dx) / (width * SupersampleSize) * 2 - 1, ((float)(y * SupersampleSize + dy) / (height * SupersampleSize) * 2 - 1) * height / width);
			float2 coordinates = screenCoords * zoom + (float2)(xCenter, yCenter);
			int i = Iterate(coordinates);
			accum += (float3)(sin(i / 17.0f) * 0.5f + 0.5f, sin(i / 19.0f) * 0.5f + 0.5f, sin(i / 23.0f) * 0.5f + 0.5f);
		}
	}
	screen[y * width + x] = (float4)(accum / (SupersampleSize * SupersampleSize), 1);
}
