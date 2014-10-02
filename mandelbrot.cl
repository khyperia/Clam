/*

CLAMSCRIPTSTART

posx = 0.0
posy = 0.0
zoom = 1.0

function derp()
end

compile("mandelbrot.cl")

function update(time)
    if iskeydown("w") then posy = posy - zoom * time end
    if iskeydown("s") then posy = posy + zoom * time end
    if iskeydown("a") then posx = posx - zoom * time end
    if iskeydown("d") then posx = posx + zoom * time end
    if iskeydown("r") then zoom = zoom / (1 + time) end
    if iskeydown("f") then zoom = zoom * (1 + time) end

    if iskeydown("p") then
        unsetkey("p")
        width = 10000
        height = 10000
        mkbuffer("screenshot", width * height * 4 * 4)
        kernel("main", width, height, "screenshot",
            {math.floor(-width / 2)}, {math.floor(-height / 2)}, {width}, {height},
            posx, posy, zoom / width)
        dlbuffer("screenshot", width)
        rmbuffer("screenshot")
    end
    
    kernel("main", -1, -1, "", derp, posx, posy, zoom / 1000)
end

CLAMSCRIPTEND

*/

#ifndef MaxIters
#define MaxIters 2048
#endif

#ifndef Bailout
#define Bailout 2
#endif

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
    float offsetX, float offsetY, float zoom)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;
    
    float2 coords = (float2)((float)(x + sx), (float)(y + sy));
    coords = coords * zoom + (float2)(offsetX, offsetY);

    float3 color = Iterate(coords);

    screen[y * width + x] = (float4)(color, 1);
}
