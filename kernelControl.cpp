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
    auto &pos = settings.Get("pos").AsVec2();
    auto &julia = settings.Get("julia").AsVec2();
    temp.posX = (float)pos.x;
    temp.posY = (float)pos.y;
    temp.zoom = (float)settings.Get("zoom").AsFloat();
    temp.juliaX = (float)julia.x;
    temp.juliaY = (float)julia.y;
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
    auto pos = settings.Get("pos").AsVec3();
    auto look = settings.Get("look").AsVec3().normalized();
    auto
        up = cross(cross(look, settings.Get("up").AsVec3()), look).normalized();
    temp.posX = (float)pos.x;
    temp.posY = (float)pos.y;
    temp.posZ = (float)pos.z;
    temp.lookX = (float)look.x;
    temp.lookY = (float)look.y;
    temp.lookZ = (float)look.z;
    temp.upX = (float)up.x;
    temp.upY = (float)up.y;
    temp.upZ = (float)up.z;
#define DefineVec3(name) do { \
        const auto& val = settings.Get(#name).AsVec3();\
        temp.name##X = val.x;\
        temp.name##Y = val.y;\
        temp.name##Z = val.z;\
    } while (0)
#define DefineFlt(name) temp.name = (float)settings.Get(#name).AsFloat()
#define DefineInt(name) temp.name = (int)settings.Get(#name).AsInt()
#define DefineBool(name) temp.name = settings.Get(#name).AsBool() ? 1 : 0
    DefineFlt(fov);
    DefineFlt(focalDistance);

    DefineFlt(Scale);
    DefineFlt(FoldingLimit);
    DefineFlt(FixedRadius2);
    DefineFlt(MinRadius2);
    DefineFlt(DeRotationAmount);
    DefineVec3(DeRotationAxis);
    DefineFlt(DofAmount);
    DefineFlt(FovAbberation);

    DefineVec3(LightPos);
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
#undef DefineVec3
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
