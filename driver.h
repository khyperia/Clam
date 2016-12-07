#pragma once

#include <functional>
#include <memory>
#include "display.h"
#include "kernelConfig.h"
#include "kernelControl.h"
#include "kernel.h"
#include "cudaContext.h"

extern bool cudaSuccessfulInit;

struct RenderType;

class RealtimeRender: public Immobile
{
    std::unique_ptr<RenderTarget> renderTarget;
    std::unique_ptr<GpuKernel> kernel;
    SettingCollection settings;
    std::vector<std::unique_ptr<KernelControl>> kernelControls;
    std::vector<std::unique_ptr<UiSetting>> uiSettings;
    Uint32 last_enqueue_time;

    static std::function<void(int *, size_t, size_t)> GpuCallback();
    void EnqueueKernel(int frame);
    std::string ConfigText();
    static void PushCallback(std::function<void(RealtimeRender &)> func);
public:
    RealtimeRender(CudaContext context,
                   const KernelConfiguration &kernel,
                   int width,
                   int height,
                   const char *fontName);
    ~RealtimeRender();
    void StartLoop(size_t queue_size);
    bool Tick();
};
