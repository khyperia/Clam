#pragma once
#define MandelboxStateSize 16
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
    float DofAmount;
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
    int MaxIters;
    float Bailout;
    float DeMultiplier;
    int RandSeedInitSteps;
    float MaxRayDist;
    int MaxRaySteps;
    int NumRayBounces;
    float QualityFirstRay;
    float QualityRestRay;
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
