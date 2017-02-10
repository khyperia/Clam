typedef void *CUdeviceptr;

#include "kernelStructs.h"

// for ide
#ifndef __constant__
#define __constant__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#define MaxIters 1000
#define Bailout 16*16
#define ColorVariance 0.2f
#define Saturation 0.4f
#define Value 0.6f
#define flt float
#define flt2 float2
#define make_flt2 make_float2
__constant__ MandelbrotCfg CfgArr[1];
#define Cfg (CfgArr[0])

static __device__ float3 HueToRGB(float hue, float saturation, float value)
{
    hue = fmod(hue, 1.0f);
    hue *= 3;
    float frac = fmod(hue, 1.0f);
    float3 color;
    switch ((int)hue)
    {
        case 0:
            color = make_float3(1 - frac, frac, 0);
            break;
        case 1:
            color = make_float3(0, 1 - frac, frac);
            break;
        case 2:
            color = make_float3(frac, 0, 1 - frac);
            break;
        default:
            color = make_float3(1, 1, 1);
            break;
    }
    color.x = sqrtf(color.x);
    color.y = sqrtf(color.y);
    color.z = sqrtf(color.z);
    saturation = value * (1 - saturation);
    color.x = color.x * (value - saturation) + saturation;
    color.y = color.y * (value - saturation) + saturation;
    color.z = color.z * (value - saturation) + saturation;
    return color;
}

static __device__ float3 Mandelbrot(flt2 z)
{
    flt2 c = Cfg.juliaEnabled ? make_flt2(Cfg.juliaX, Cfg.juliaY) : z;
    int i = 0;
    while (i < MaxIters && z.x * z.x + z.y * z.y < Bailout)
    {
        z = make_flt2(z.x * z.x - z.y * z.y + c.x, 2 * z.x * z.y + c.y);
        i++;
    }
    if (i == MaxIters)
    {
        return make_float3(54 / 255.0f, 70 / 255.0f, 93 / 255.0f);
    }
    float mu = i - log2(log(z.x * z.x + z.y * z.y));
    float x = mu * ColorVariance;
    return HueToRGB(
        x * ColorVariance, Saturation, Value
    );
}

static __device__ unsigned int PackPixel(
    float3 pixel
)
{
    pixel.x *= 255;
    pixel.y *= 255;
    pixel.z *= 255;
    if (pixel.x < 0)
    {
        pixel.x = 0;
    }
    if (pixel.x > 255)
    {
        pixel.x = 255;
    }
    if (pixel.y < 0)
    {
        pixel.y = 0;
    }
    if (pixel.y > 255)
    {
        pixel.y = 255;
    }
    if (pixel.z < 0)
    {
        pixel.z = 0;
    }
    if (pixel.z > 255)
    {
        pixel.z = 255;
    }
    return (255 << 24) | ((int)pixel.x << 16) | ((int)pixel.y << 8) | ((int)pixel.z);
}

extern "C" __global__ void kern(
    unsigned int *__restrict__ screenPixels,
    int screenX,
    int screenY,
    int width,
    int height,
    int frame
)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    flt fx = (flt)(x + screenX);
    flt fy = (flt)(y + screenY);
    float mulBy = Cfg.zoom * 2 / (width + height);
    //float mulBy = 1.0 * 2 / (width + height);

    fx *= mulBy;
    fy *= mulBy;
    fx += Cfg.posX;
    fy += Cfg.posY;
    float3 color = Mandelbrot(make_flt2(fx, fy));
    screenPixels[y * width + x] = PackPixel(color);
}
