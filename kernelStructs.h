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

#define MandelboxStateSize 80

struct MandelboxCfg
{
    CUdeviceptr scratch;
    CUdeviceptr randbuf;
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

    float Scale;
    float FoldingLimit;
    float FixedRadius2;
    float MinRadius2;
    float DeRotationAmount;
    float DeRotationAxisX;
    float DeRotationAxisY;
    float DeRotationAxisZ;
    float DofAmount;
    float FovAbberation;

    float LightPosX;
    float LightPosY;
    float LightPosZ;
    float LightSize;

    int WhiteClamp;

    float LightBrightnessHue;
    float LightBrightnessSat;
    float LightBrightnessVal;

    float AmbientBrightnessHue;
    float AmbientBrightnessSat;
    float AmbientBrightnessVal;

    float ReflectHue;
    float ReflectSat;
    float ReflectVal;

    int MaxIters;
    float Bailout;
    float DeMultiplier;
    int RandSeedInitSteps;
    float MaxRayDist;
    int MaxRaySteps;
    int NumRayBounces;
    float QualityFirstRay;
    float QualityRestRay;
    int ItersPerKernel;
};

struct MandelbrotCfg
{
    float posX;
    float posY;
    float zoom;
    float juliaX;
    float juliaY;
    int juliaEnabled;
};
