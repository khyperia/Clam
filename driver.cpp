#include "driver.h"
#include "lib_init.h"

std::function<void(int *, size_t, size_t)> RealtimeRender::GpuCallback()
{
    const auto result = [](int *data, size_t width, size_t height)
    {
        const auto cb = [data, width, height](RealtimeRender &self)
        {
            self.renderTarget->Blit(BlitData(data, (int)width, (int)height),
                                    self.ConfigText());
            self.EnqueueKernel(0);
        };
        RealtimeRender::PushCallback(cb);
    };
    return result;
}

RealtimeRender::RealtimeRender(CudaContext context,
                               const KernelConfiguration &kernel,
                               int width,
                               int height,
                               const char *fontName)
    : renderTarget(make_unique<SdlWindow>(100, 100, width, height, fontName)),
      kernel(make_unique<GpuKernel>(std::move(context),
                                    GpuCallback(),
                                    kernel.KernelData(),
                                    kernel.KernelLength())),
      settings(std::move(kernel.Settings())),
      kernelControls(std::move(kernel.Controls(*this->kernel))),
      uiSettings(std::move(kernel.UiSettings()))
{
    IncrementSdlUsage();
}

RealtimeRender::~RealtimeRender()
{
    DecrementSdlUsage();
}

std::string RealtimeRender::ConfigText()
{
    return "Hello, world!\nHey look a newline!";
}

void RealtimeRender::EnqueueKernel(int frame)
{
    size_t width, height;
    if (!renderTarget->RequestSize(&width, &height))
    {
        throw std::runtime_error("Realtime renderer must have render size hint");
    }
    auto current_ticks = SDL_GetTicks();
    float elapsed_seconds = (current_ticks - last_enqueue_time) / 1000.0f;
    last_enqueue_time = current_ticks;
    for (auto &uiSetting : uiSettings)
    {
        uiSetting->Integrate(settings, elapsed_seconds);
    }
    for (const auto &control: kernelControls)
    {
        control->SetFrom(settings);
    }
    kernel->Run(0, 0, (int)width, (int)height, frame, true);
}

void RealtimeRender::StartLoop(size_t queue_size)
{
    for (size_t i = 0; i < queue_size; i++)
    {
        EnqueueKernel(0);
    }
}

void RealtimeRender::PushCallback(std::function<void(RealtimeRender & )> func)
{
    SDL_Event event;
    event.type = SDL_USEREVENT;
    event.user.data1 =
        make_unique<std::function<void(RealtimeRender &)>>(func).release();
    SDL_PushEvent(&event);
}

bool RealtimeRender::Tick()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT)
        {
            return false;
        }
        if (event.type == SDL_USEREVENT)
        {
            std::function<void(RealtimeRender &)> *ptr =
                (std::function<void(RealtimeRender &)> *)event.user.data1;
            std::unique_ptr<std::function<void(RealtimeRender &)>> func(ptr);
            (*func)(*this);
        }
        for (auto &uiSetting : uiSettings)
        {
            uiSetting->Input(settings, event);
        }
    }
    return true;
}
