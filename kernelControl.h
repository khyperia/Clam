#pragma once

#include "kernelSetting.h"
#include "kernel.h"

class KernelControl : public Immobile
{
protected:
    GpuKernelVar &kernelVariable;
public:
    KernelControl(GpuKernelVar &kernelVariable);
    virtual ~KernelControl();
    virtual bool SetFrom(
        const SettingCollection &settings, CudaContext &context, size_t width, size_t height
    ) = 0;
};

class MandelbrotKernelControl : public KernelControl
{
public:
    MandelbrotKernelControl(GpuKernelVar &kernelVariable);
    ~MandelbrotKernelControl() override;
    bool SetFrom(
        const SettingCollection &settings, CudaContext &context, size_t width, size_t height
    ) override;
};

struct MandelboxCfg;

class MandelboxKernelControl : public KernelControl
{
    size_t old_width;
    size_t old_height;
    CuMem<char> scratch_buffer;
    CuMem<uint64_t> rand_buffer;
    std::unique_ptr<MandelboxCfg> old_state;
public:
    MandelboxKernelControl(GpuKernelVar &kernelVariable);
    ~MandelboxKernelControl() override;
    bool SetFrom(
        const SettingCollection &settings, CudaContext &context, size_t width, size_t height
    ) override;
};
