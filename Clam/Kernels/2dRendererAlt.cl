__kernel void Main(__global float4* screen, int width, int height, float xCenter, float yCenter, float zoom)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x >= width || y >= height)
		return;

	if (x == width / 4 || y == height / 2 && x < width / 2)
	{
		screen[y * width + x] = (float4)(1, 0, 0, 1);
		return;
	}

	float2 screenCoords = (float2)((float)x / width * 2 - 1, ((float)y / height * 2 - 1) * height / width);
	int i;
	if (screenCoords.x < 0)
	{
		screenCoords.x += 0.5f;
		float2 coordinates = screenCoords * zoom + (float2)(xCenter, yCenter);
		i = Iterate(coordinates);
	}
	else
	{
		screenCoords.x -= 0.5f;
		float2 coordinates = screenCoords * 3; // * zoom + (float2)(xCenter, yCenter);
		i = IterateAlt(coordinates, (float2)(xCenter, yCenter));
	}
	screen[y * width + x] = (float4)(sin(i / 17.0f) * 0.5f + 0.5f, sin(i / 19.0f) * 0.5f + 0.5f, sin(i / 23.0f) * 0.5f + 0.5f, 1);
}
