#ifndef MaxIters
#define MaxIters 256
#endif

#ifndef Bailout
#define Bailout 8
#endif

uint MWC64X(ulong *state)
{
    uint c=(*state)>>32, x=(*state)&0xFFFFFFFF;
    *state = x*((ulong)4294883355U) + c;
    return x^c;
}

float Rand(ulong* seed)
{
    return (float)MWC64X(seed) / UINT_MAX;
}

float3 GetColor(float i)
{
    return (float3)(sin(i / 17.0) * 0.5 + 0.5,
                    sin(i / 19.0) * 0.5 + 0.5,
                    sin(i / 23.0) * 0.5 + 0.5);
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

__kernel void main(__global float4* screen,
    int sx, int sy, int width, int height,
    float offsetX, float offsetY, float zoom,
    float juliaX, float juliaY,
    int frame)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;

    ulong seed = sx + width * sy + width * height * frame;

    float2 coords = (float2)((float)(x + sx), (float)(y + sy))
        + (float2)(Rand(&seed), Rand(&seed)) * 1.1;
    coords = coords * zoom + (float2)(offsetX, offsetY);

    float3 color = juliaX == 0 && juliaY == 0 ? Iterate(coords) : IterateAlt(coords, (float2)(juliaX, juliaY));

    float3 prev = screen[y * width + x].xyz;
    if (frame != 0)
        prev = (color + prev * frame) / (frame + 1);
    else
        prev = color;
    screen[y * width + x] = (float4)(prev, 1);
}
