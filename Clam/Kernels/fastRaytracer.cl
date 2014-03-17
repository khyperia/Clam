#ifndef Fov
#define Fov 1.0
#endif

#ifndef MaxRayIters
#define MaxRayIters 64
#endif

__kernel void Main(__global float4* screen, int screenWidth, int width, int height, float4 position, float4 lookat, float4 updir)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x >= width || y >= height)
		return;
	
	float2 screenCoords = (float2)((float)x / width * 2 - 1, ((float)y / height * 2 - 1) * height / width);

	float3 pos = position.xyz;
	float3 direction = RayDir(lookat.xyz, updir.xyz, screenCoords, Fov);

	float totalDistance = 0.0;
	int i;
	for (i = 0; i < MaxRayIters; i++)
	{
		float distance = De(pos + direction * totalDistance);
		if (distance * width / 10 < totalDistance)
			break;
		totalDistance += distance;
	}
	float value = (float)i / MaxRayIters;
	int screenIndex = screenWidth ? (y - get_global_offset(1)) * screenWidth + (x - get_global_offset(0)) : y * width + x;
	screen[screenIndex] = (float4)(value, value, value, 1.0);
}
