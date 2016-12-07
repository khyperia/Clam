#pragma once

#if 0
#ifndef __constant__
#define __constant__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#endif

struct GpuCameraSettings
{
    float posX;
    float posY;
    float posZ;
    float lookX;
    float lookY;
    float lookZ;
    float upX;
    float upY;
    float upZ;
    float fov;
    float focalDistance;
};

struct Gpu2dCameraSettings
{
    float posX;
    float posY;
    float zoom;
};

struct JuliaBrotSettings
{
    float juliaX;
    float juliaY;
    int juliaEnabled;
};
