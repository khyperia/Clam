#pragma once

#define Mandelbox2StateSize (22*sizeof(float))

struct MandelboxCfg
{
    float Scale;
    float FoldingLimit;
    float FixedRadius2;
    float MinRadius2;
    float DeRotation;
    float ColorSharpness;
    float Saturation;
    float HueVariance;
    float Reflectivity;
    float DofAmount;
    float FovAbberation;

    float LightPosX;
    float LightPosY;
    float LightPosZ;
    float LightSize;

    float ColorBiasR;
    float ColorBiasG;
    float ColorBiasB;
    int WhiteClamp;
    float BrightThresh;

    float SpecularHighlightAmount;
    float SpecularHighlightSize;

    float FogDensity;

    float LightBrightnessAmount;
    float LightBrightnessCenter;
    float LightBrightnessWidth;

    float AmbientBrightnessAmount;
    float AmbientBrightnessCenter;
    float AmbientBrightnessWidth;

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
    x.DeRotation = 0;
    x.ColorSharpness = 1.0f;
    x.Saturation = 0.5f;
    x.HueVariance = 1.0f;
    x.Reflectivity = 1.0f;
    x.DofAmount = 0.005f;
    x.FovAbberation = 0.01f;
    x.SpecularHighlightAmount = 1.0f;
    x.SpecularHighlightSize = 0.5f;
    x.LightBrightnessAmount = 100;
    x.LightBrightnessCenter = 0.25f;
    x.LightBrightnessWidth = 0.5f;
    x.AmbientBrightnessAmount = 4;
    x.AmbientBrightnessCenter = 0.75f;
    x.AmbientBrightnessWidth = 0.25f;
    x.LightPosX = 3;
    x.LightPosY = 3;
    x.LightPosZ = 3;
    x.LightSize = 0.1f;
    x.FogDensity = 0.0001f;
    x.ColorBiasR = 0;
    x.ColorBiasG = 0;
    x.ColorBiasB = 0;
    x.WhiteClamp = 0;
    x.BrightThresh = 8;
    x.MaxIters = 64;
    x.Bailout = 1024;
    x.DeMultiplier = 0.95f;
    x.RandSeedInitSteps = 128;
    x.MaxRayDist = 16;
    x.MaxRaySteps = 256;
    x.NumRayBounces = 3;
    x.QualityFirstRay = 2;
    x.QualityRestRay = 64;
    x.ItersPerKernel = 5;
    return x;
}
