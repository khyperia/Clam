#include "cudaContext.h"

int CudaContext::currentContext = -1;

void CudaContext::CheckCall(CUresult callResult)
{
    if (callResult != CUDA_SUCCESS)
    {
        const char *errstr;
        if (cuGetErrorString(callResult, &errstr) == CUDA_SUCCESS)
        {
            throw std::runtime_error(
                "CUDA error (" + tostring(callResult) + "): " + errstr);
        }
        else if (cuGetErrorName(callResult, &errstr) == CUDA_SUCCESS)
        {
            throw std::runtime_error(
                "CUDA error " + tostring(callResult) + " = " + errstr);
        }
        else
        {
            throw std::runtime_error(
                "CUDA error " + tostring(callResult) + " (no name)");
        }
    }
}

void CudaContext::RunWithoutContext(CUresult callResult)
{
    CheckCall(callResult);
}

void CudaContext::Run(CUresult callResult) const
{
    if (currentContext != deviceIndex)
    {
        throw std::runtime_error(
            "Current context not correct, call CudaContext::SetCurrent() before this call. This context: "
                + tostring(deviceIndex) + ", active context: "
                + tostring(currentContext));
    }
    CheckCall(callResult);
}

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
    Run(cuCtxCreate(&context, CU_CTX_BLOCKING_SYNC, device));
}

CudaContext::~CudaContext()
{
    if (context != 0)
    {
        cuCtxDestroy(context);
    }
    if (currentContext == deviceIndex)
    {
        currentContext = -1;
    }
    deviceIndex = -1;
    device = 0;
    context = 0;
}

CudaContext::CudaContext(CudaContext && other)
{
    deviceIndex = other.deviceIndex;
    device = other.device;
    context = other.context;
    other.deviceIndex = -1;
    other.device = 0;
    other.context = 0;
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
