#include "kernelControl.h"

#include "kernelStructs.h"

KernelControl::KernelControl(GpuKernelVar &kernelVariable)
    : kernelVariable(kernelVariable)
{
}

KernelControl::~KernelControl()
{
}

MandelbrotKernelControl::MandelbrotKernelControl(GpuKernelVar &kernelVariable)
    : KernelControl(kernelVariable)
{
}

void MandelbrotKernelControl::SetFrom(const SettingCollection &settings)
{
    MandelbrotCfg temp;
    temp.posX = settings.Get("posx").AsFloat();
    temp.posY = settings.Get("posy").AsFloat();
    temp.zoom = settings.Get("zoom").AsFloat();
    temp.juliaX = settings.Get("juliax").AsFloat();
    temp.juliaY = settings.Get("juliay").AsFloat();
    temp.juliaEnabled = settings.Get("juliaenabled").AsBool() ? 1 : 0;
    kernelVariable.Set(temp);
}

MandelboxKernelControl::MandelboxKernelControl(GpuKernelVar &kernelVariable)
    : KernelControl(kernelVariable)
{
}

void MandelboxKernelControl::SetFrom(const SettingCollection &settings)
{
    MandelboxCfg temp;
#define DefineFlt(name) temp.name = (float)settings.Get(#name).AsFloat()
#define DefineInt(name) temp.name = (int)settings.Get(#name).AsInt()
#define DefineBool(name) temp.name = settings.Get(#name).AsBool() ? 1 : 0
    DefineFlt(posX);
    DefineFlt(posY);
    DefineFlt(posZ);
    DefineFlt(lookX);
    DefineFlt(lookY);
    DefineFlt(lookZ);
    DefineFlt(upX);
    DefineFlt(upY);
    DefineFlt(upZ);
    DefineFlt(fov);
    DefineFlt(focalDistance);

    DefineFlt(Scale);
    DefineFlt(FoldingLimit);
    DefineFlt(FixedRadius2);
    DefineFlt(MinRadius2);
    DefineFlt(DeRotationAmount);
    DefineFlt(DeRotationAxisX);
    DefineFlt(DeRotationAxisY);
    DefineFlt(DeRotationAxisZ);
    DefineFlt(DofAmount);
    DefineFlt(FovAbberation);

    DefineFlt(LightPosX);
    DefineFlt(LightPosY);
    DefineFlt(LightPosZ);
    DefineFlt(LightSize);

    DefineBool(WhiteClamp);

    DefineFlt(LightBrightnessHue);
    DefineFlt(LightBrightnessSat);
    DefineFlt(LightBrightnessVal);

    DefineFlt(AmbientBrightnessHue);
    DefineFlt(AmbientBrightnessSat);
    DefineFlt(AmbientBrightnessVal);

    DefineFlt(ReflectHue);
    DefineFlt(ReflectSat);
    DefineFlt(ReflectVal);

    DefineInt(MaxIters);
    DefineFlt(Bailout);
    DefineFlt(DeMultiplier);
    DefineInt(RandSeedInitSteps);
    DefineFlt(MaxRayDist);
    DefineInt(MaxRaySteps);
    DefineInt(NumRayBounces);
    DefineFlt(QualityFirstRay);
    DefineFlt(QualityRestRay);
    DefineInt(ItersPerKernel);
#undef DefineFlt
#undef DefineInt
#undef DefineBool
    kernelVariable.Set(temp);
}
