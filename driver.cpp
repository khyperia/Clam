#include <fstream>
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

RealtimeRender::RealtimeRender(
    CudaContext context,
    const KernelConfiguration &kernel,
    int width,
    int height,
    const char *fontName
) : renderTarget(make_unique<SdlWindow>(100, 100, width, height, fontName)),
    kernel(
        make_unique<GpuKernel>(
            std::move(context), GpuCallback(), kernel.KernelData(), kernel.KernelLength())),
    settings(std::move(kernel.Settings())),
    kernelControls(std::move(kernel.Controls(*this->kernel))),
    uiSettings(std::move(kernel.UiSettings())),
    last_enqueue_time(0),
    fpsAverage(0),
    take_screenshot(false)
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
    kernel->Run((int)width / -2, (int)height / -2, (int)width, (int)height, changed ? 0 : 1, true);
}

void RealtimeRender::StartLoop(size_t queue_size)
{
    for (size_t i = 0; i < queue_size; i++)
    {
        EnqueueKernel();
    }
}

void RealtimeRender::PushCallback(std::function<void(class RealtimeRender &)> func)
{
    SDL_Event event;
    event.type = SDL_USEREVENT;
    event.user.data1 = make_unique<std::function<void(RealtimeRender &)>>(func).release();
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
                std::cout << "Didn't load state, settings.clam3 not found" << std::endl;
            }
        }
        if (event.key.keysym.scancode == SDL_SCANCODE_V)
        {
            auto saved = settings.Save();
            std::ofstream out("movie.clam3", std::ofstream::out | std::ofstream::app);
            out << saved << "---" << std::endl;
            std::cout << "Saved keyframe" << std::endl;
        }
        if (event.key.keysym.scancode == SDL_SCANCODE_B)
        {
            if (std::remove("movie.clam3"))
            {
                std::cout << "Unable to delete movie file" << std::endl;
            }
            else
            {
                std::cout << "Deleted keyframe file" << std::endl;
            }
        }
    }
}

void RealtimeRender::Run()
{
    StartLoop(2);
    SDL_Event event;
    while (SDL_WaitEvent(&event))
    {
        if (event.type == SDL_QUIT)
        {
            break;
        }
        if (event.type == SDL_USEREVENT)
        {
            std::function<void(RealtimeRender &)> *ptr = (std::function<void(RealtimeRender &)> *)event
                .user
                .data1;
            std::unique_ptr<std::function<void(RealtimeRender &)>> func;
            func.reset(ptr);
            (*func)(*this);
        }
        for (auto &uiSetting : uiSettings)
        {
            uiSetting->Input(settings, event);
        }
        DriverInput(event);
    }
}

HeadlessRender::HeadlessRender(
    CudaContext context,
    const KernelConfiguration &kernel,
    size_t width,
    size_t height,
    int num_frames,
    const std::string settings_file,
    const std::string filename
) : kernel(
    make_unique<GpuKernel>(
        std::move(context),
        GpuCallback(std::move(filename)),
        kernel.KernelData(),
        kernel.KernelLength())),
    settings(std::move(kernel.Settings())),
    kernelControls(std::move(kernel.Controls(*this->kernel))),
    num_frames(num_frames),
    width(width),
    height(height)
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
            "Couldn't render: " + settings_file + " not found"
        );
    }
    for (const auto &control : kernelControls)
    {
        control->SetFrom(settings, this->kernel->Context(), width, height);
    }
}

std::function<void(int *, size_t, size_t)> HeadlessRender::GpuCallback(std::string filename)
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
    TimeEstimate timer;
    for (int frame = 0; frame < num_frames; frame++)
    {
        kernel->Run((int)width / -2,
            (int)height / -2,
            (int)width,
            (int)height,
            frame,
            frame == num_frames - 1
        );
        if (frame % sync_every == sync_every - 1)
        {
            kernel->SyncStream();
            std::cout << timer.Mark(frame, num_frames) << std::endl;
        }
    }
    kernel->SyncStream();
    std::cout << "Done." << std::endl;
}

MovieRender::MovieRender(
    CudaContext context,
    const KernelConfiguration &kernel,
    size_t width,
    size_t height,
    int num_iters,
    int num_frames,
    bool loop,
    const std::string settings_file,
    const std::string base_filename
) : filename(std::move(std::make_shared<std::string>(std::move(base_filename)))),
    kernel(
        make_unique<GpuKernel>(
            std::move(context), GpuCallback(filename), kernel.KernelData(), kernel.KernelLength())),
    template_settings(std::move(kernel.Settings())),
    settings(),
    kernelControls(std::move(kernel.Controls(*this->kernel))),
    num_iters(num_iters),
    num_frames(num_frames),
    loop(loop),
    width(width),
    height(height)
{
    std::ifstream in(settings_file);
    if (in)
    {
        std::ostringstream contents;
        std::string line;
        while (true)
        {
            bool end = !std::getline(in, line);
            if (end || line == "---")
            {
                std::string value = contents.str();
                if (!value.empty())
                {
                    SettingCollection setting = template_settings.Clone();
                    setting.Load(value);
                    settings.push_back(std::move(setting));
                }
                contents.str("");
                contents.clear();
            }
            else if (!line.empty())
            {
                contents << line << std::endl;
            }
            if (end)
            {
                break;
            }
        }
        if (settings.size() < 2)
        {
            throw std::runtime_error(
                "Couldn't render: " + settings_file + " no keyframes in file"
            );
        }
    }
    else
    {
        throw std::runtime_error(
            "Couldn't render: " + settings_file + " not found"
        );
    }
}

std::function<void(int *, size_t, size_t)>
MovieRender::GpuCallback(std::shared_ptr<std::string> filename)
{
    const auto result = [filename](int *data, size_t width, size_t height)
    {
        BlitData blitData(data, (int)width, (int)height);
        FileTarget::Screenshot(*filename, blitData);
    };
    return result;
}

MovieRender::~MovieRender()
{
}

void MovieRender::Run()
{
    TimeEstimate timer;
    for (int frame = 0; frame < num_frames; frame++)
    {
        *filename = "movie." + tostring(frame) + ".bmp";
        double time = (double)frame / num_frames * (settings.size() - (loop ? 0 : 1));
        int keyframe = (int)time;
        time = time - keyframe;
        int t0 = keyframe - 1;
        int t1 = keyframe;
        int t2 = keyframe + 1;
        int t3 = keyframe + 2;
        if (t0 < 0)
        {
            t0 = loop ? t0 + (int)settings.size() : 0;
        }
        if (t2 >= (int)settings.size())
        {
            t2 = loop ? t2 - (int)settings.size() : (int)settings.size() - 1;
        }
        if (t3 >= (int)settings.size())
        {
            t3 = loop ? t3 - (int)settings.size() : (int)settings.size() - 2;
        }
        SettingCollection this_frame = SettingCollection::Interpolate(
            settings[t0], settings[t1], settings[t2], settings[t3], time
        );
        for (const auto &control : kernelControls)
        {
            control->SetFrom(this_frame, this->kernel->Context(), width, height);
        }
        for (int iter = 0; iter < num_iters; iter++)
        {
            kernel->Run((int)width / -2,
                (int)height / -2,
                (int)width,
                (int)height,
                iter,
                iter == num_iters - 1
            );
        }
        kernel->SyncStream();
        std::cout << timer.Mark(frame, num_frames) << std::endl;
    }
}
