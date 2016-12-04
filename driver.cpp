#include "driver.h"
#include "option.h"
#include "lib_init.h"

/*
static SDL_Rect GetWindowPos()
{
    std::string winpos = WindowPos();
    if (winpos.empty())
    {
        std::cout
            << "Window position wasn't specified. Defaulting to 500x500+100+100"
            << std::endl;
        winpos = "500x500+100+100";
    }
    SDL_Rect windowPos;
    if (sscanf(winpos.c_str(),
               "%dx%d%d%d",
               &windowPos.w,
               &windowPos.h,
               &windowPos.x,
               &windowPos.y) != 4)
    {
        throw std::runtime_error("Window position was in an incorrect format");
    }
    return windowPos;
}

struct NoRenderType: public RenderType
{
    SDL_Window *window;
    TTF_Font *font;

    NoRenderType(SDL_Rect windowPos)
        : RenderType()
    {
        WindowCreate(&window,
                     &font,
                     windowPos.x,
                     windowPos.y,
                     windowPos.w,
                     windowPos.h);
    }

    ~NoRenderType()
    {
        WindowDestroy(window, font);
    }

    virtual void Boot(Kernel *, const CudaContext context)
    {
        EnqueueBlitData(BlitData(), context);
    }

    virtual void BlitFrame(BlitData, Kernel *kernel, const CudaContext context)
    {
        kernel->UpdateNoRender();
        SDL_Surface *surface = SDL_GetWindowSurface(window);
        SDL_FillRect(surface, NULL, 0);
        SDL_Surface *conf = kernel->Configure(font);
        if (conf)
        {
            SDL_BlitSurface(conf, NULL, surface, NULL);
            SDL_FreeSurface(conf);
        }
        SDL_UpdateWindowSurface(window);

        // dummy data just so we get called again
        EnqueueBlitData(BlitData(), context);
    }
};

struct SyncBlitData
{
    BlitData blitData;
    const CudaContext context;

    SyncBlitData(BlitData blitData, const CudaContext context)
        : blitData(blitData), context(context)
    {
    }
};

static void EnqueueSyncBlitDataHelper(CUstream, CUresult, void *userData)
{
    SyncBlitData *blitData = (SyncBlitData *)userData;
    BlitData data = blitData->blitData;
    const CudaContext context = blitData->context;
    delete blitData;
    EnqueueBlitData(data, context);
}

static void EnqueueSyncBlitData(CUstream stream,
                                BlitData blitData,
                                const CudaContext context)
{
    SyncBlitData *syncBlitData = new SyncBlitData(blitData, context);
    context.Run(cuStreamAddCallback(stream,
                                    EnqueueSyncBlitDataHelper,
                                    syncBlitData,
                                    0));
}

struct CpuRenderType: public RenderType
{
    SDL_Window *window;
    TTF_Font *font;
    std::vector<BlitData> oldBlitData;

    CpuRenderType(SDL_Rect windowPos, size_t numContexts)
        : RenderType(), oldBlitData()
    {
        for (size_t i = 0; i < numContexts; i++)
        {
            oldBlitData.push_back(BlitData());
        }
        WindowCreate(&window,
                     &font,
                     windowPos.x,
                     windowPos.y,
                     windowPos.w,
                     windowPos.h);
    }

    ~CpuRenderType()
    {
        WindowDestroy(window, font);
    }

    void Blit(BlitData pixbuf, SDL_Surface *surface) const
    {
        if (SDL_LockSurface(surface))
        {
            throw std::runtime_error("Could not lock SDL surface");
        }
        if (surface->format->BytesPerPixel != 4)
        {
            throw std::runtime_error("Window surface bytes/pixel != 4");
        }
        size_t pixbuf_size = (size_t)pixbuf.width * pixbuf.height;
        size_t surface_size = (size_t)surface->w * surface->h;
        size_t size = pixbuf_size < surface_size ? pixbuf_size : surface_size;
        size_t byte_size = size * surface->format->BytesPerPixel;
        memcpy(surface->pixels, pixbuf.data, size);
        SDL_UnlockSurface(surface);
    }

    void DrawUI(Kernel *kern, TTF_Font *font, SDL_Surface *surface) const
    {
        SDL_Surface *conf = kern->Configure(font);
        if (conf)
        {
            SDL_BlitSurface(conf, NULL, surface, NULL);
            SDL_FreeSurface(conf);
        }
    }

    virtual void Boot(Kernel *kernel, const CudaContext context)
    {
        EnqueueFrame(kernel, context);
    }

    void EnqueueFrame(Kernel *kernel, const CudaContext context)
    {
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        if (width <= 0 || height <= 0)
        {
            throw std::runtime_error("Invalid SDL window size");
        }
        if (oldBlitData[context.Index()].width != width
            || oldBlitData[context.Index()].height != height)
        {
            EnqueueCuMemFreeHost(oldBlitData[context.Index()].data, context);
            oldBlitData[context.Index()].data = NULL;
            oldBlitData[context.Index()].width = width;
            oldBlitData[context.Index()].height = height;
        }
        if (!oldBlitData[context.Index()].data)
        {
            int32_t **data = &oldBlitData[context.Index()].data;
            size_t size = (size_t)width * height * sizeof(int) * 4;
            context.Run(cuMemAllocHost((void **)data, size));
        }
        kernel->RenderInto(oldBlitData[context.Index()].data,
                           (size_t)oldBlitData[context.Index()].width,
                           (size_t)oldBlitData[context.Index()].height,
                           context);
        EnqueueSyncBlitData(kernel->Stream(context),
                            oldBlitData[context.Index()],
                            context);
    }

    virtual void
    BlitFrame(BlitData pixbuf, Kernel *kernel, const CudaContext context)
    {
        // this is above the follwing so that the GPU gets started as we're blitting
        EnqueueFrame(kernel, context);

        SDL_Surface *surface = SDL_GetWindowSurface(window);
        Blit(pixbuf, surface);
        if (IsUserInput())
        {
            DrawUI(kernel, font, surface);
        }
        SDL_UpdateWindowSurface(window);
    }
};

struct HeadlessRenderType: public RenderType
{
    const int numFrames;
    std::vector<int> currentFrames;
    const int numTimes;
    std::vector<int> currentTimeByContext;
    int currentTime;
    const std::vector<CudaContext> contexts;
    std::vector<BlitData> mBlitData;

    HeadlessRenderType(int numFrames,
                       int numTimes,
                       int width,
                       int height,
                       const std::vector<CudaContext> contexts)
        : numFrames(numFrames), currentFrames(), numTimes(numTimes),
          currentTime(0), contexts(contexts), mBlitData()
    {
        for (size_t i = 0; i < contexts.size(); i++)
        {
            contexts[i].SetCurrent();
            BlitData blitData;
            blitData.width = width;
            blitData.height = height;
            int32_t **data = &blitData.data;
            size_t size = (size_t)width * height * sizeof(int) * 4;
            contexts[i].Run(cuMemAllocHost((void **)data, size));
            mBlitData.push_back(blitData);
            currentTimeByContext.push_back(currentTime);
            if (numTimes > 1)
            {
                currentTime++;
            }
        }
        if (numTimes <= 1)
        {
            currentTime++;
        }
    }

    ~HeadlessRenderType()
    {
        for (size_t i = 0; i < contexts.size(); i++)
        {
            contexts[i].SetCurrent();
            contexts[i].Run(cuMemFreeHost(mBlitData[i].data));
        }
    }

    std::string RenderstateFilename(Kernel *kernel,
                                    int currentTime,
                                    const CudaContext context) const
    {
        std::ostringstream builder;
        builder << kernel->Name();
        int shiftx, shifty;
        if (RenderOffset(&shiftx, &shifty))
        {
            builder << "." << shiftx << "x" << shifty;
        }
        builder << ".renderstate";
        if (numTimes > 1)
        {
            builder << ".t" << currentTime;
        }
        else if (contexts.size() != 1)
        {
            builder << ".gpu" << context.Index();
        }
        builder << ".clam3";
        return builder.str();
    }

    void LoadAnimationState(Kernel *kernel) const
    {
        std::cout << "Loading animation keyframes" << std::endl;
        kernel->LoadAnimation();
    }

    void
    LoadFileState(Kernel *kernel, int time, const CudaContext context) const
    {
        std::string filename = RenderstateFilename(kernel, time, context);
        try
        {
            // Loads *.renderstate.clam3 (the previously-left-off render)
            StateSync *sync = NewFileStateSync(filename.c_str(), true);
            kernel->RecvState(sync, true, context);
            delete sync;
            std::cout << "Loaded intermediate state from " << filename
                      << std::endl;
        }
        catch (const std::exception &ex)
        {
            // If that fails, try *.clam3 (the saved state parameters)
            std::cout << "Didn't load intermediate state from " << filename
                      << ": " << ex.what() << std::endl
                      << "Trying initial headless state instead" << std::endl;
            StateSync *sync =
                NewFileStateSync((kernel->Name() + ".clam3").c_str(), true);
            kernel->RecvState(sync, false, context);
            delete sync;
        }
    }

    void SaveProgress(Kernel *kernel,
                      int currentTime,
                      const CudaContext context) const
    {
        std::string
            filename = RenderstateFilename(kernel, currentTime, context);
        StateSync *sync = NewFileStateSync(filename.c_str(), false);
        kernel->SendState(sync, true, context);
        delete sync;
        std::cout << "Saved intermediate progress to " << filename << std::endl;
    }

    virtual void Boot(Kernel *kernel, const CudaContext context)
    {
        if (currentFrames.size() != (size_t)context.Index())
        {
            throw std::runtime_error("Headless contexts not added in order"
                                         " (this is an implicit "
                                         "dependency in code)");
        }
        currentFrames.push_back(0);
        const BlitData &data = mBlitData[context.Index()];
        kernel->Resize((size_t)data.width, (size_t)data.height, context);
        EnqueueMany(kernel, context);
    }

    void EnqueueMany(Kernel *kernel, const CudaContext context)
    {
        const int syncEveryNFrames = 32;
        for (int i = 0; i < syncEveryNFrames; i++)
        {
            if (EnqueueSingle(kernel, mBlitData[context.Index()], context))
            {
                break;
            }
        }
        EnqueueSyncBlitData(kernel->Stream(context),
                            mBlitData[context.Index()],
                            context);
    }

    // returns true if current frame can't be added to anymore
    virtual bool
    EnqueueSingle(Kernel *kernel, BlitData blitData, const CudaContext context)
    {
        // this is one of the implicit dependencies mentioned above
        if (currentTimeByContext[context.Index()] == 0
            && currentFrames[context.Index()] == 0)
        {
            if (numTimes > 1)
            {
                LoadAnimationState(kernel);
            }
            else
            {
                LoadFileState(kernel,
                              currentTimeByContext[context.Index()],
                              context);
                int frame = kernel->GetFrame(context);
                if (frame > 0)
                {
                    currentFrames[context.Index()] = frame;
                }
            }
            // note we can't refactor SetFramed to here, due to possibly just loading a renderstate.clam3
        }
        if (currentFrames[context.Index()] == 0)
        {
            if (numTimes > 1)
            {
                double time =
                    (double)currentTimeByContext[context.Index()] / numTimes;
                //time = -std::cos(time * 6.28318530718) * 0.5 + 0.5;
                kernel->SetTime(time, false, context); // boolean is `wrap`
            }
            kernel->SetFramed(true, context);
        }
        kernel->RenderInto(
            currentFrames[context.Index()] == numFrames - 1 ? blitData.data
                                                            : NULL,
            (size_t)blitData.width,
            (size_t)blitData.height,
            context);
        currentFrames[context.Index()]++;
        return currentFrames[context.Index()] >= numFrames;
    }

    void WriteFrame(BlitData pixbuf,
                    Kernel *kernel,
                    int currentTime,
                    const CudaContext context) const
    {
        SDL_Surface *surface = SDL_CreateRGBSurface(0,
                                                    pixbuf.width,
                                                    pixbuf.height,
                                                    4 * 8,
                                                    255 << 16,
                                                    255 << 8,
                                                    255,
                                                    0);
        if (SDL_LockSurface(surface))
        {
            throw std::runtime_error("Could not lock temp buffer surface");
        }
        memcpy(surface->pixels,
               pixbuf.data,
               (size_t)pixbuf.width * pixbuf.height
                   * surface->format->BytesPerPixel);
        SDL_UnlockSurface(surface);
        std::string
            filename = RenderstateFilename(kernel, currentTime, context);
        filename += ".bmp";
        SDL_SaveBMP(surface, filename.c_str());
        std::cout << "Saved image '" << filename << "'" << std::endl;
        SDL_FreeSurface(surface);
    }

    virtual void
    BlitFrame(BlitData pixbuf, Kernel *kernel, const CudaContext context)
    {
        if (DoSaveProgress() && numTimes <= 1)
        {
            SaveProgress(kernel,
                         currentTimeByContext[context.Index()],
                         context);
        }
        if (numTimes <= 1)
        {
            std::cout << numFrames - currentFrames[context.Index()]
                      << " frames left in this image, "
                      << numTimes - currentTimeByContext[context.Index()]
                      << " images left" << std::endl;
        }
        if (currentFrames[context.Index()] >= numFrames)
        {
            WriteFrame(pixbuf,
                       kernel,
                       currentTimeByContext[context.Index()],
                       context);
            currentFrames[context.Index()] = 0;
            currentTimeByContext[context.Index()] = currentTime;
            currentTime++;
        }
        if (currentTimeByContext[context.Index()] >= numTimes)
        {
            currentFrames[context.Index()] = -1;
            for (size_t i = 0; i < currentFrames.size(); i++)
            {
                if (currentFrames[i] != -1)
                {
                    return;
                }
            }
            // only once all are set to -1 (done) do we quit
            SDL_Event quitEvent;
            quitEvent.type = SDL_QUIT;
            SDL_PushEvent(&quitEvent);
        }
        else
        {
            EnqueueMany(kernel, context);
        }
    }
};

static CUcontext InitCuda(int deviceNum)
{
    if (!cudaSuccessfulInit)
    {
        throw std::runtime_error(
            "CUDA device init failure while in compute mode");
    }
    int maxDev = 0;
    if (cuDeviceGetCount(&maxDev) != CUDA_SUCCESS)
    {
        throw std::runtime_error("Could not get number of CUDA devices");
    }
    if (maxDev <= 0)
    {
        throw std::runtime_error(
            "cuDeviceGetCount returned zero, is an NVIDIA GPU present on the system?");
    }
    if (deviceNum < 0 || deviceNum >= maxDev)
    {
        throw std::runtime_error("Invalid device number " + tostring(deviceNum)
                                     + ": must be less than "
                                     + tostring(maxDev));
    }
    CUdevice cuDevice;
    if (cuDeviceGet(&cuDevice, deviceNum) != CUDA_SUCCESS)
    {
        throw std::runtime_error(
            "Could not get CUDA device " + tostring(deviceNum));
    }
    {
        char name[128];
        if (cuDeviceGetName(name, sizeof(name) - 1, cuDevice) != CUDA_SUCCESS)
        {
            throw std::runtime_error(
                "Could not get CUDA device name for device number "
                    + tostring(deviceNum));
        }
        std::cout << "Using device (" << deviceNum << " of " << maxDev << "): "
                  << name << std::endl;
    }
    CUcontext result;
    if (cuCtxCreate(&result, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice)
        != CUDA_SUCCESS)
    {
        throw std::runtime_error("Could not create CUDA context");
    }
    return result;
}

const std::vector<CudaContext> InitCudaContexts()
{
    std::vector<CudaContext> cuContexts;
    if (!IsCompute())
        return cuContexts;
    std::vector<int> deviceNums = CudaDeviceNums();
    for (size_t i = 0; i < deviceNums.size(); i++)
    {
        cuContexts.push_back(CudaContext(InitCuda(deviceNums[i]), (int)i));
    }
    return cuContexts;
}

Driver::Driver()
    : cuContexts(InitCudaContexts()), connection()
{
    bool isCompute = IsCompute();
    int numHeadlessTimes;
    int headless = Headless(&numHeadlessTimes);

    kernel = new Kernel(KernelName(), cuContexts);
    if (headless == 0)
    {
        SDL_Rect windowPos = GetWindowPos();
        renderTarget = new SdlWindow(windowPos.x,
                                     windowPos.y,
                                     windowPos.w,
                                     windowPos.h,
                                     FontName().c_str());
    }
    else
    {
        renderTarget = new FileTarget(kernel->Name());
    }
}

Driver::~Driver()
{
    delete renderType;
    delete kernel;
    for (size_t i = 0; i < cuContexts.size(); i++)
    {
        cuContexts[i].SetCurrent();
        cuContexts[i].Run(cuCtxDestroy(cuContexts[i].UnderlyingContext()));
    }
}

void Driver::Tick()
{
    Uint32 currentTime = SDL_GetTicks();
    double time = (currentTime - lastTickTime) / 1000.0;
    lastTickTime = currentTime;

    // double weight = 1 / (fpsAverage + 1);
    // fpsAverage = (timePassed + fpsAverage * weight) / (weight + 1);
    // timeSinceLastTitle += timePassed;
    // while (timeSinceLastTitle > 1.0)
    // {
    //     SDL_Delay(1);
    //     SDL_SetWindowTitle(window, ("Clam3 - " + tostring(1 / fpsAverage) + " fps").c_str());
    //     timeSinceLastTitle--;
    // }

    if (IsUserInput())
    {
        kernel->Integrate(time);
    }
    if (connection.IsSyncing())
    {
        if (connection.Sync(kernel))
        {
            SDL_Event event;
            event.type = SDL_QUIT;
            SDL_PushEvent(&event);
        }
    }
}

void Driver::MainLoop()
{
    SDL_Event event;
    for (size_t i = 0; i < cuContexts.size(); i++)
    {
        cuContexts[i].SetCurrent();
        renderType->Boot(kernel, cuContexts[i]);
    }
    if (dynamic_cast<NoRenderType *>(renderType) && cuContexts.size() == 0)
    {
        renderType->Boot(kernel, CudaContext::Invalid);
    }
    bool isUserInput = IsUserInput();
    while (true)
    {
        if (!SDL_WaitEvent(&event))
        {
            throw std::runtime_error(SDL_GetError());
        }
        if (event.type == SDL_QUIT)
        {
            break;
        }
        if (isUserInput)
        {
            kernel->UserInput(event);
        }
        if (event.type == SDL_USEREVENT)
        {
            int context = event.user.code;
            void
            (*func)(Driver *, const CudaContext, void *) = (void (*)(Driver *,
                                                                     const CudaContext,
                                                                     void *))event
                .user.data1;
            void *param = event.user.data2;
            const CudaContext thisContext =
                context == -1 ? CudaContext::Invalid : cuContexts[context];
            if (thisContext.IsValid())
                thisContext.SetCurrent();
            func(this, thisContext, param);
        }
    }
}

void EnqueueSdlEvent(void (*funcPtr)(Driver *, const CudaContext, void *),
                     void *data,
                     const CudaContext context)
{
    SDL_Event event;
    event.type = SDL_USEREVENT;
    event.user.code = context.IsValid() ? context.Index() : -1;
    event.user.data1 = (void *)funcPtr;
    event.user.data2 = data;
    SDL_PushEvent(&event);
}

static void
EnqueueCuMemFreeHelper(Driver *, const CudaContext context, void *hostPtr)
{
    context.Run(cuMemFreeHost(hostPtr));
}

void EnqueueCuMemFreeHost(void *hostPtr, const CudaContext context)
{
    EnqueueSdlEvent(EnqueueCuMemFreeHelper, hostPtr, context);
}

void Driver::BlitImmediate(const BlitData blitData, const CudaContext context)
{
    Tick(); // well, where else am I going to put it?
    renderType->BlitFrame(blitData, kernel, context);
}

static void
EnqueueBlitDataHelper(Driver *driver, const CudaContext context, void *blitData)
{
    BlitData *data = (BlitData *)blitData;
    BlitData temp = *data;
    delete data;
    driver->BlitImmediate(temp, context);
}

void EnqueueBlitData(const BlitData blitData, const CudaContext context)
{
    EnqueueSdlEvent(EnqueueBlitDataHelper, new BlitData(blitData), context);
}
*/

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
        RealtimeRender::PushCallbackImpl(cb);
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
      isUserInput(IsUserInput())
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
    kernel->Run(0, 0, (int)width, (int)height, frame, true);
}

void RealtimeRender::StartLoop(size_t queue_size)
{
    for (size_t i = 0; i < queue_size; i++)
    {
        EnqueueKernel(0);
    }
}

void
RealtimeRender::PushCallbackImpl(std::function<void(RealtimeRender & )> func)
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
        else if (isUserInput)
        {
            // TODO
            //kernel->UserInput(event);
        }
    }
    return true;
}
