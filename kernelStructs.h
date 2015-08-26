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
