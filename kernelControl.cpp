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
    Gpu2dCameraSettings temp;
    temp.posX = (float)settings.Get("posx").AsFloat();
    temp.posY = (float)settings.Get("posy").AsFloat();
    temp.zoom = (float)settings.Get("zoom").AsFloat();
    kernelVariable.Set(temp);
}
