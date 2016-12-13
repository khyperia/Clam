#pragma once

#include "kernelSetting.h"
#include "kernel.h"

class KernelControl: public Immobile
{
protected:
    GpuKernelVar &kernelVariable;
public:
    KernelControl(GpuKernelVar &kernelVariable);
    ~KernelControl();

    virtual void SetFrom(const SettingCollection &settings) = 0;
};

class MandelbrotKernelControl: public KernelControl
{
public:
    MandelbrotKernelControl(GpuKernelVar &kernelVariable);
    void SetFrom(const SettingCollection &settings) override;
};

class MandelboxKernelControl: public KernelControl
{
public:
    MandelboxKernelControl(GpuKernelVar &kernelVariable);
    void SetFrom(const SettingCollection &settings) override;
};
