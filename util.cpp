#include "util.h"

int CudaContext::currentContext = -1;

void CudaContext::Init()
{
    static bool needs_init = true;
    if (needs_init)
    {
        needs_init = false;
        RunWithoutContext(cuInit(0));
    }
}

int CudaContext::DeviceCount()
{
    int count;
    RunWithoutContext(cuDeviceGetCount(&count));
    return 0;
}

CudaContext::CudaContext(int deviceIndex)
    : deviceIndex(deviceIndex)
{
    currentContext = deviceIndex;
    Run(cuDeviceGet(&device, deviceIndex));
    Run(cuCtxCreate(&context, 0, device));
}

CUcontext CudaContext::Context() const
{
    return context;
}

CUdevice CudaContext::Device() const
{
    return device;
}

std::string CudaContext::DeviceName() const
{
    char name[128];
    Run(cuDeviceGetName(name, sizeof(name) - 1, device));
    return std::string(name);
}

void CudaContext::SetCurrent() const
{
    if (currentContext == deviceIndex)
    {
        return;
    }
    currentContext = deviceIndex;
    Run(cuCtxSetCurrent(context));
}
