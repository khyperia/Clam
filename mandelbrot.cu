#include "kernelStructs.h"

#define MaxIters 1000
#define Bailout 16*16
#define ColorVariance 0.1f
#define MultipR 7.0f
#define MultipG 9.0f
#define MultipB 13.0f
#define OffsetR 0.0f
#define OffsetG 0.0f
#define OffsetB 0.0f
#define Saturation 0.2f
#define SubtractBrightness 0.6f

__constant__ Gpu2dCameraSettings CameraArr[1];

#define Camera (CameraArr[0])

__constant__ JuliaBrotSettings JuliaArr[1];

#define Julia (JuliaArr[0])

static __device__ float3 Mandelbrot(float2 z)
{
    //float2 c = Julia.juliaEnabled ? make_float2(Julia.juliaX, Julia.juliaY) : z;
    float2 c = z;

    int i = 0;
    while (i < MaxIters && z.x * z.x + z.y * z.y < Bailout)
    {
        z = make_float2(z.x * z.x - z.y * z.y + c.x, 2 * z.x * z.y + c.y);
        i++;
    }
    if (i == MaxIters)
    {
        return make_float3(54 / 255.0f, 70 / 255.0f, 93 / 255.0f);
    }
    float mu = i - log2(log(z.x * z.x + z.y * z.y));
    float x = mu * ColorVariance;
    float r = sinf(x * MultipR + OffsetR) * 0.5f + 0.5f;
    float g = sinf(x * MultipG + OffsetG) * 0.5f + 0.5f;
    float b = sinf(x * MultipB + OffsetB) * 0.5f + 0.5f;
    r = 1 - (1 - r) * Saturation - SubtractBrightness;
    g = 1 - (1 - g) * Saturation - SubtractBrightness;
    b = 1 - (1 - b) * Saturation - SubtractBrightness;
    return make_float3(r, g, b);
}

static __device__ unsigned int PackPixel(float3 pixel)
{
    pixel.x *= 255;
    pixel.y *= 255;
    pixel.z *= 255;
    if (pixel.x < 0)
        pixel.x = 0;
    if (pixel.x > 255)
        pixel.x = 255;
    if (pixel.y < 0)
        pixel.y = 0;
    if (pixel.y > 255)
        pixel.y = 255;
    if (pixel.z < 0)
        pixel.z = 0;
    if (pixel.z > 255)
        pixel.z = 255;
    return (255 << 24) | ((int)pixel.x << 16) | ((int)pixel.y << 8)
        | ((int)pixel.z);
}

extern "C" __global__ void kern(unsigned int *__restrict__ screenPixels,
                                int screenX,
                                int screenY,
                                int width,
                                int height,
                                int frame)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    float fx = (float)(x + screenX);
    float fy = (float)(y + screenY);

    float mulBy = Camera.zoom * 2 / (width + height);
    //float mulBy = 1.0 * 2 / (width + height);

    fx *= mulBy;
    fy *= mulBy;

    fx += Camera.posX;
    fy += Camera.posY;

    float3 color = Mandelbrot(make_float2(fx, fy));
    screenPixels[y * width + x] = PackPixel(color);
}
