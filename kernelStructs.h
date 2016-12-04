#pragma once

#ifndef __constant__
#define __constant__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
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

    bool operator==(const GpuCameraSettings &right)
    {
        return memcmp(this, &right, sizeof(GpuCameraSettings)) == 0;
    }

    bool operator!=(const GpuCameraSettings &right)
    {
        return memcmp(this, &right, sizeof(GpuCameraSettings)) != 0;
    }
};

struct Gpu2dCameraSettings
{
    float posX;
    float posY;
    float zoom;

    bool operator==(const Gpu2dCameraSettings &right)
    {
        return memcmp(this, &right, sizeof(Gpu2dCameraSettings)) == 0;
    }

    bool operator!=(const Gpu2dCameraSettings &right)
    {
        return memcmp(this, &right, sizeof(Gpu2dCameraSettings)) != 0;
    }
};

struct JuliaBrotSettings
{
    float juliaX;
    float juliaY;
    int juliaEnabled;

    bool operator==(const JuliaBrotSettings &right)
    {
        return memcmp(this, &right, sizeof(JuliaBrotSettings)) == 0;
    }

    bool operator!=(const JuliaBrotSettings &right)
    {
        return memcmp(this, &right, sizeof(JuliaBrotSettings)) != 0;
    }
};
