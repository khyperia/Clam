#pragma once

#include <string.h>

#define MandelboxStateSize (22*sizeof(float))

struct MandelboxCfg
{
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

    bool operator==(const MandelboxCfg &right)
    {
        return memcmp(this, &right, sizeof(MandelboxCfg)) == 0;
    }

    bool operator!=(const MandelboxCfg &right)
    {
        return memcmp(this, &right, sizeof(MandelboxCfg)) != 0;
    }
};

MandelboxCfg MandelboxDefault()
{
    MandelboxCfg x;
    x.Scale = -2.0f;
    x.FoldingLimit = 1.0f;
    x.FixedRadius2 = 1.0f;
    x.MinRadius2 = 0.25f;
    x.DeRotationAmount = 0;
    x.DeRotationAxisX = 1;
    x.DeRotationAxisY = 1;
    x.DeRotationAxisZ = 1;
    x.DofAmount = 0.005f;
    x.FovAbberation = 0.01f;
    x.LightBrightnessHue = 0.025f;
    x.LightBrightnessSat = 0.5f;
    x.LightBrightnessVal = 8.0f;
    x.AmbientBrightnessHue = 0.525f;
    x.AmbientBrightnessSat = 0.25f;
    x.AmbientBrightnessVal = 1.0f;
    x.ReflectHue = 0.75f;
    x.ReflectSat = 0.01f;
    x.ReflectVal = 1.0f;
    x.LightPosX = 3;
    x.LightPosY = 3;
    x.LightPosZ = 3;
    x.LightSize = 1.0f;
    x.WhiteClamp = 0;
    x.MaxIters = 64;
    x.Bailout = 1024;
    x.DeMultiplier = 0.95f;
    x.RandSeedInitSteps = 128;
    x.MaxRayDist = 16;
    x.MaxRaySteps = 256;
    x.NumRayBounces = 3;
    x.QualityFirstRay = 2;
    x.QualityRestRay = 64;
    x.ItersPerKernel = 8;
    return x;
}
