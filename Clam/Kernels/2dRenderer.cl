__kernel void Main(__global float4* screen, int width, int height, float xCenter, float yCenter, float zoom)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x >= width || y >= height)
		return;

	float2 screenCoords = (float2)((float)x / width * 2 - 1, ((float)y / height * 2 - 1) * height / width);
	float2 coordinates = screenCoords * zoom + (float2)(xCenter, yCenter);
	int i = Iterate(coordinates);
	screen[y * width + x] = (float4)(sin(i / 17.0f) * 0.5f + 0.5f, sin(i / 19.0f) * 0.5f + 0.5f, sin(i / 23.0f) * 0.5f + 0.5f, 1);
}
