#include <fstream>
#include <chrono>
#include <iomanip>
#include "driver.h"
#include "lib_init.h"

std::function<void(int *, size_t, size_t)> RealtimeRender::GpuCallback()
{
    const auto result = [](int *data, size_t width, size_t height)
    {
        const auto cb = [data, width, height](RealtimeRender &self)
        {
            self.EnqueueKernel();
            BlitData blitData(data, (int)width, (int)height);
            self.renderTarget->Blit(blitData, self.ConfigText());
            if (self.take_screenshot)
            {
                self.take_screenshot = false;
                FileTarget::Screenshot("screenshot.bmp", blitData);
            }
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
      uiSettings(std::move(kernel.UiSettings())), last_enqueue_time(0),
      fpsAverage(0), take_screenshot(false)
{
    IncrementSdlUsage();
}

RealtimeRender::~RealtimeRender()
{
    DecrementSdlUsage();
}

void RealtimeRender::UpdateFps(double elapsed_seconds)
{
    const double weight = 1 / (fpsAverage + 1);
    fpsAverage = (elapsed_seconds + fpsAverage * weight) / (weight + 1);
}

std::string RealtimeRender::ConfigText()
{
    std::string result;
    for (const auto &control : uiSettings)
    {
        auto conf = control->Describe(settings);
        if (!conf.empty())
        {
            result += conf;
        }
    }
    return tostring(1 / fpsAverage) + " fps\n" + result;
}

void RealtimeRender::EnqueueKernel()
{
    size_t width, height;
    if (!renderTarget->RequestSize(&width, &height))
    {
        throw std::runtime_error("Realtime renderer must have render size hint");
    }
    auto current_ticks = SDL_GetTicks();
    double elapsed_seconds = (current_ticks - last_enqueue_time) / 1000.0;
    last_enqueue_time = current_ticks;
    UpdateFps(elapsed_seconds);
    for (auto &uiSetting : uiSettings)
    {
        uiSetting->Integrate(settings, elapsed_seconds);
    }
    bool changed = false;
    for (const auto &control: kernelControls)
    {
        changed |= control->SetFrom(settings, kernel->Context(), width, height);
    }
    kernel->Run((int)width / -2,
                (int)height / -2,
                (int)width,
                (int)height,
                changed ? 0 : 1,
                true);
}

void RealtimeRender::StartLoop(size_t queue_size)
{
    for (size_t i = 0; i < queue_size; i++)
    {
        EnqueueKernel();
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

void RealtimeRender::DriverInput(SDL_Event event)
{
    if (event.type == SDL_KEYDOWN)
    {
        if (event.key.keysym.scancode == SDL_SCANCODE_P)
        {
            take_screenshot = true;
        }
        if (event.key.keysym.scancode == SDL_SCANCODE_T)
        {
            auto saved = settings.Save();
            std::ofstream out("settings.clam3");
            out << saved;
            std::cout << "Saved state" << std::endl;
        }
        if (event.key.keysym.scancode == SDL_SCANCODE_Y)
        {
            std::ifstream in("settings.clam3");
            if (in)
            {
                std::ostringstream contents;
                contents << in.rdbuf();
                settings.Load(contents.str());
                std::cout << "Loaded state" << std::endl;
            }
            else
            {
                std::cout << "Didn't load state, settings.clam3 not found"
                          << std::endl;
            }
        }
    }
}

void RealtimeRender::Run()
{
    StartLoop(1);
    SDL_Event event;
    while (SDL_WaitEvent(&event))
    {
        if (event.type == SDL_QUIT)
        {
            break;
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
        DriverInput(event);
    }
}

HeadlessRender::HeadlessRender(CudaContext context,
                               const KernelConfiguration &kernel,
                               size_t width,
                               size_t height,
                               int num_frames,
                               const std::string settings_file,
                               const std::string filename)
    : kernel(make_unique<GpuKernel>(std::move(context),
                                    GpuCallback(std::move(filename)),
                                    kernel.KernelData(),
                                    kernel.KernelLength())),
      settings(std::move(kernel.Settings())),
      kernelControls(std::move(kernel.Controls(*this->kernel))),
      num_frames(num_frames), width(width), height(height)
{
    std::ifstream in(settings_file);
    if (in)
    {
        std::ostringstream contents;
        contents << in.rdbuf();
        settings.Load(contents.str());
    }
    else
    {
        throw std::runtime_error(
            "Couldn't render: " + settings_file + " not found");
    }
    for (const auto &control : kernelControls)
    {
        control->SetFrom(settings, this->kernel->Context(), width, height);
    }
}

std::function<void(int *, size_t, size_t)>
HeadlessRender::GpuCallback(std::string filename)
{
    const auto result = [filename](int *data, size_t width, size_t height)
    {
        BlitData blitData(data, (int)width, (int)height);
        std::cout << "Saving: " << filename << std::endl;
        FileTarget::Screenshot(filename, blitData);
    };
    return result;
}

HeadlessRender::~HeadlessRender()
{
}

void HeadlessRender::Run()
{
    const int sync_every = 32;
    auto start = std::chrono::system_clock::now();
    for (int frame = 0; frame < num_frames; frame++)
    {
        kernel->Run((int)width / -2,
                    (int)height / -2,
                    (int)width,
                    (int)height,
                    frame,
                    frame == num_frames - 1);
        if (frame % sync_every == sync_every - 1)
        {
            kernel->SyncStream();
            auto now = std::chrono::system_clock::now();
            auto frame1 = frame + 1;
            auto elapsed = now - start;
            auto ticks_per = elapsed / frame1;
            auto frames_left = num_frames - frame1;
            auto time_left = ticks_per * frames_left;

            auto elapsed_sec =
                std::chrono::duration_cast<std::chrono::seconds>(elapsed)
                    .count();
            auto elapsed_min = elapsed_sec / 60;
            elapsed_sec -= elapsed_min * 60;
            auto time_left_sec =
                std::chrono::duration_cast<std::chrono::seconds>(time_left)
                    .count();
            auto time_left_min = time_left_sec / 60;
            time_left_sec -= time_left_min * 60;

            std::cout << frame1 << "/" << num_frames << " ("
                      << (int)((100.0 * frame1) / num_frames) << "%), "
                      << std::setfill('0') << elapsed_min << ":"
                      << std::setw(2) << elapsed_sec << " elapsed, "
                      << time_left_min << ":" << std::setw(2)
                      << time_left_sec << " left" << std::setfill(' ')
                      << std::setw(0) << std::endl;
        }
    }
    kernel->SyncStream();
    std::cout << "Done." << std::endl;
}
