#pragma once

#include <functional>
#include <memory>
#include "display.h"
#include "kernel.h"

extern bool cudaSuccessfulInit;

struct RenderType;

class RealtimeRender
{
    std::unique_ptr<RenderTarget> renderTarget;
    std::unique_ptr<GpuKernel> kernel;
    bool isUserInput;

    static std::function<void(int *, size_t, size_t)> GpuCallback();
    void EnqueueKernel(int frame);
    std::string ConfigText();
    static void PushCallbackImpl(std::function<void(RealtimeRender &)> func);
public:
    RealtimeRender(CudaContext context,
                   const KernelConfiguration &kernel,
                   int width,
                   int height,
                   const char *fontName);
    ~RealtimeRender();
    void StartLoop(size_t queue_size);
    template<typename F>
    void PushCallback(F f)
    {
        this->PushCallbackImpl(make_unique(f));
    }
    bool Tick();
};

/*
class Driver
{
    Kernel *kernel;
    RenderTarget *renderTarget;
    const std::vector<CudaContext> cuContexts;
    Connection connection;
    Uint32 lastTickTime;
public:
    Driver();

    ~Driver();

    void MainLoop();

    void BlitImmediate(const BlitData blitData, const CudaContext context);

    void Tick();
};

void EnqueueCuMemFreeHost(void *hostPtr, const CudaContext context);

void EnqueueBlitData(BlitData blitData, const CudaContext context);
*/
