#ifndef MaxIters
#define MaxIters 2048
#endif

#ifndef Bailout
#define Bailout 2
#endif

float3 GetColor(float i)
{
	return (float3)(sin(i / 17.0) * 0.5 + 0.5, sin(i / 19.0) * 0.5 + 0.5, sin(i / 23.0) * 0.5 + 0.5);
}

float ComputeSmooth(float2 last)
{
	return 1 + log2(log((float)Bailout) / log(length(last)));
}

float3 IterateAlt(float2 z, float2 c)
{
    for (int it = 0; it < MaxIters; it++)
    {
	float x2 = z.x * z.x;
	float y2 = z.y * z.y;
	if (x2 + y2 > Bailout * Bailout)
		return GetColor(ComputeSmooth(z) + it);
	z = (float2)(x2 - y2, 2 * z.x * z.y) + c;
    }
    return (float3)(0,0,0);
}

float3 Iterate(float2 z)
{
    return IterateAlt(z, z);
}

__kernel void main(__global float4* screen, int width, int height, float offsetX, float offsetY, float zoom)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;
    
    float2 coords = (float2)((float)x / width, (float)y / height) * 2.0f - (float2)(1.0f);
    coords.y *= (float)height / width;
    coords = coords * zoom + (float2)(offsetX, offsetY);

    float3 color = Iterate(coords);

    screen[y * width + x] = (float4)(color, 1);
}
