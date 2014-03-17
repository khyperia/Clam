#ifndef SupersampleSize
#define SupersampleSize 1
#endif

__kernel void Main(__global float4* screen, int screenWidth, int width, int height, flt xCenter, flt yCenter, flt zoom)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x >= width || y >= height)
		return;
		
	int screenIndex = screenWidth ? (y - get_global_offset(1)) * screenWidth + (x - get_global_offset(0)) : y * width + x;
	if (x == width / 4 || y == height / 2 && x < width / 2)
	{
		screen[screenIndex] = (float4)(1, 0, 0, 1);
		return;
	}

	float3 color = (float3)(0,0,0);
	for (int dx = 0; dx < SupersampleSize; dx++)
	{
		for (int dy = 0; dy < SupersampleSize; dy++)
		{
			flt2 screenCoords = (flt2)((flt)(x * SupersampleSize + dx) / (width * SupersampleSize) * 2 - 1, ((flt)(y * SupersampleSize + dy) / (height * SupersampleSize) * 2 - 1) * height / width);
			if (screenCoords.x < 0)
			{
				screenCoords.x += 0.5;
				flt2 coordinates = screenCoords * zoom + (flt2)(xCenter, yCenter);
				color += Iterate(coordinates);
			}
			else
			{
				screenCoords.x -= 0.5;
				flt2 coordinates = screenCoords * 3; // * zoom + (flt2)(xCenter, yCenter);
				color += IterateAlt(coordinates, (flt2)(xCenter, yCenter));
			}
		}
	}
	screen[screenIndex] = (float4)(color, 1);
}
