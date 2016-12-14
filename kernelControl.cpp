#include <cstring>
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

MandelbrotKernelControl::~MandelbrotKernelControl()
{
}

bool MandelbrotKernelControl::SetFrom(const SettingCollection &settings,
                                      CudaContext &,
                                      size_t,
                                      size_t)
{
    MandelbrotCfg temp;
    temp.posX = (float)settings.Get("posx").AsFloat();
    temp.posY = (float)settings.Get("posy").AsFloat();
    temp.zoom = (float)settings.Get("zoom").AsFloat();
    temp.juliaX = (float)settings.Get("juliax").AsFloat();
    temp.juliaY = (float)settings.Get("juliay").AsFloat();
    temp.juliaEnabled = settings.Get("juliaenabled").AsBool() ? 1 : 0;
    kernelVariable.Set(temp);
    return true;
}

MandelboxKernelControl::MandelboxKernelControl(GpuKernelVar &kernelVariable)
    : KernelControl(kernelVariable), old_width(0), old_height(0),
      scratch_buffer(), rand_buffer()
{
}

MandelboxKernelControl::~MandelboxKernelControl()
{
}

bool MandelboxKernelControl::SetFrom(const SettingCollection &settings,
                                     CudaContext &context,
                                     size_t width,
                                     size_t height)
{
    bool changed = false;
    if (old_width != width || old_height != height || !scratch_buffer()
        || !rand_buffer())
    {
        changed = true;
        old_width = width;
        old_height = height;
        scratch_buffer.~CuMem();
        rand_buffer.~CuMem();
        new(&scratch_buffer) CuMem<char>(context,
                                         width * height * MandelboxStateSize);
        new(&rand_buffer) CuMem<uint64_t>(context, width * height);
    }
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
    if (!old_state)
    {
        changed = true;
        old_state = make_unique<MandelboxCfg>();
        *old_state = temp;
    }
    else if (memcmp(old_state.get(), &temp, sizeof(temp)))
    {
        changed = true;
        *old_state = temp;
    }
    temp.scratch = scratch_buffer();
    temp.randbuf = rand_buffer();
    kernelVariable.Set(temp);
    return changed;
}
