#pragma once

#include <functional>
#include <memory>
#include "display.h"
#include "kernelConfig.h"
#include "kernelControl.h"
#include "kernel.h"

class RealtimeRender: public Immobile
{
    std::unique_ptr<RenderTarget> renderTarget;
    std::unique_ptr<GpuKernel> kernel;
    SettingCollection settings;
    std::vector<std::unique_ptr<KernelControl>> kernelControls;
    std::vector<std::unique_ptr<UiSetting>> uiSettings;
    Uint32 last_enqueue_time;
    double fpsAverage;
    bool take_screenshot;

    static std::function<void(int *, size_t, size_t)> GpuCallback();
    void EnqueueKernel();
    void StartLoop(size_t queue_size);
    void UpdateFps(double elapsed_seconds);
    std::string ConfigText();
    static void PushCallback(std::function<void(RealtimeRender &)> func);
    void DriverInput(SDL_Event event);
public:
    RealtimeRender(CudaContext context,
                   const KernelConfiguration &kernel,
                   int width,
                   int height,
                   const char *fontName);
    ~RealtimeRender();
    void Run();
};

class HeadlessRender: public Immobile
{
    std::unique_ptr<GpuKernel> kernel;
    SettingCollection settings;
    std::vector<std::unique_ptr<KernelControl>> kernelControls;
    int num_frames;
    size_t width, height;

    static std::function<void(int *, size_t, size_t)>
    GpuCallback(std::string filename);
public:
    HeadlessRender(CudaContext context,
                   const KernelConfiguration &kernel,
                   size_t width,
                   size_t height,
                   int num_frames,
                   const std::string settings_file,
                   const std::string filename);
    ~HeadlessRender();
    void Run();
};
