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
    if (!old_state)
    {
        changed = true;
        old_state = make_unique<MandelboxCfg>();
        *old_state = temp;
    }
    auto pos = settings.Get("pos").AsVec3();
    auto look = settings.Get("look").AsVec3().normalized();
    auto up = cross(cross(look, settings.Get("up").AsVec3()), look).normalized();
    changed |= (float)pos.x != old_state->posX;
    changed |= (float)pos.y != old_state->posY;
    changed |= (float)pos.z != old_state->posZ;
    changed |= (float)look.x != old_state->lookX;
    changed |= (float)look.y != old_state->lookY;
    changed |= (float)look.z != old_state->lookZ;
    changed |= (float)up.x != old_state->upX;
    changed |= (float)up.y != old_state->upY;
    changed |= (float)up.z != old_state->upZ;
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
        changed |= (float)val.x != old_state->name##X; \
        changed |= (float)val.y != old_state->name##Y; \
        changed |= (float)val.z != old_state->name##Z; \
        temp.name##X = (float)val.x;\
        temp.name##Y = (float)val.y;\
        temp.name##Z = (float)val.z;\
    } while (0)
#define DefineFlt(name) do { \
        const auto& val = (float)settings.Get(#name).AsFloat(); \
        changed |= val != old_state->name; \
        temp.name = val; \
    } while (0)
#define DefineInt(name) do { \
        const auto& val = settings.Get(#name).AsInt(); \
        changed |= val != old_state->name; \
        temp.name = val; \
    } while (0)
#define DefineBool(name) do { \
        const auto& val = settings.Get(#name).AsBool(); \
        changed |= val != (old_state->name != 0); \
        temp.name = val ? 1 : 0; \
    } while (0)
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
    *old_state = temp;
    temp.scratch = scratch_buffer();
    temp.randbuf = rand_buffer();
    kernelVariable.Set(temp);
    return changed;
}
