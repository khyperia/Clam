#pragma once

#include "movable.h"
#include "util.h"
#include <cuda.h>
#include <stdexcept>

class CudaContext : public NoClone
{
    int deviceIndex;
    CUdevice device;
    CUcontext context;
    static int currentContext;
    static void CheckCall(CUresult callResult);
public:
    static void Init();
    static void RunWithoutContext(CUresult callResult);
    static int DeviceCount();
    CudaContext(int deviceIndex);
    ~CudaContext();
    CudaContext(CudaContext &&);
    void SetCurrent() const;
    CUcontext Context() const;
    CUdevice Device() const;
    std::string DeviceName() const;
    void Run(CUresult callResult) const;
};
