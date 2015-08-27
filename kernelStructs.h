#pragma once

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

    bool operator==(const GpuCameraSettings& right)
    {
        return memcmp(this, &right, sizeof(GpuCameraSettings)) == 0;
    }

    bool operator!=(const GpuCameraSettings& right)
    {
        return memcmp(this, &right, sizeof(GpuCameraSettings)) != 0;
    }
};

struct Gpu2dCameraSettings
{
    float posX;
    float posY;
    float zoom;

    bool operator==(const Gpu2dCameraSettings& right)
    {
        return memcmp(this, &right, sizeof(Gpu2dCameraSettings)) == 0;
    }

    bool operator!=(const Gpu2dCameraSettings& right)
    {
        return memcmp(this, &right, sizeof(Gpu2dCameraSettings)) != 0;
    }
};
