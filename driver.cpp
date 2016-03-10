#include "driver.h"
#include "option.h"

bool cudaSuccessfulInit = false;

struct CudaInitClass
{
    CudaInitClass()
    {
        CUresult result = cuInit(0);
        if (result == CUDA_SUCCESS)
        {
            cudaSuccessfulInit = true;
        }
        else
        {
            std::cout << "cuInit failed\n";
        }
    }
} cudaInitInstance;

struct RenderType
{
    RenderType()
    {
    }

    virtual ~RenderType()
    {
    };

    virtual void Boot(Kernel *kernel) = 0;

    virtual void BlitFrame(BlitData pixbuf, Kernel *kernel) = 0;
};

static SDL_Rect GetWindowPos()
{
    std::string winpos = WindowPos();
    if (winpos.empty())
    {
        puts("Window position wasn't specified. Defaulting to 500x500+100+100");
        winpos = "500x500+100+100";
    }
    SDL_Rect windowPos;
    if (sscanf(winpos.c_str(), "%dx%d%d%d", &windowPos.w, &windowPos.h, &windowPos.x, &windowPos.y) != 4)
    {
        throw std::runtime_error("Window position was in an incorrect format");
    }
    return windowPos;
}

struct NoRenderType : public RenderType
{
    SDL_Window *window;
    TTF_Font *font;

    NoRenderType(SDL_Rect windowPos) : RenderType()
    {
        WindowCreate(&window, &font, windowPos.x, windowPos.y, windowPos.w, windowPos.h);
    }

    ~NoRenderType()
    {
        WindowDestroy(window, font);
    }

    virtual void Boot(Kernel *)
    {
        EnqueueBlitData(BlitData());
    }

    virtual void BlitFrame(BlitData, Kernel *kernel)
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
        EnqueueBlitData(BlitData());
    }
};

static void EnqueueSyncBlitDataHelper(CUstream, CUresult, void *userData)
{
    BlitData *blitData = (BlitData *)userData;
    BlitData data = *blitData;
    delete blitData;
    EnqueueBlitData(data);
}

static void EnqueueSyncBlitData(CUstream stream, BlitData blitData)
{
    HandleCu(cuStreamAddCallback(stream, EnqueueSyncBlitDataHelper, new BlitData(blitData), 0));
}

struct CpuRenderType : public RenderType
{
    SDL_Window *window;
    TTF_Font *font;
    BlitData oldBlitData;

    CpuRenderType(SDL_Rect windowPos) : RenderType(), oldBlitData()
    {
        oldBlitData.data = NULL;
        WindowCreate(&window, &font, windowPos.x, windowPos.y, windowPos.w, windowPos.h);
    }

    ~CpuRenderType()
    {
        WindowDestroy(window, font);
    }

    void Blit(BlitData pixbuf, SDL_Surface *surface)
    {
        if (SDL_LockSurface(surface))
        {
            throw std::runtime_error("Could not lock SDL surface");
        }
        if (surface->format->BytesPerPixel != 4)
        {
            throw std::runtime_error("Window surface bytes/pixel != 4");
        }
        size_t pixbuf_size = (size_t)pixbuf.width * (size_t)pixbuf.height * surface->format->BytesPerPixel;
        size_t surface_size = (size_t)surface->w * surface->h * surface->format->BytesPerPixel;
        size_t size = pixbuf_size < surface_size ? pixbuf_size : surface_size;
        memcpy(surface->pixels, pixbuf.data, size);
        SDL_UnlockSurface(surface);
    }

    void DrawUI(Kernel *kern, TTF_Font *font, SDL_Surface *surface)
    {
        SDL_Surface *conf = kern->Configure(font);
        if (conf)
        {
            SDL_BlitSurface(conf, NULL, surface, NULL);
            SDL_FreeSurface(conf);
        }
    }

    virtual void Boot(Kernel *kernel)
    {
        EnqueueFrame(kernel);
    }

    void EnqueueFrame(Kernel *kernel)
    {
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        if (width <= 0 || height <= 0)
        {
            throw std::runtime_error("Invalid SDL window size");
        }
        if (oldBlitData.width != width || oldBlitData.height != height)
        {
            EnqueueCuMemFreeHost(oldBlitData.data);
            oldBlitData.data = NULL;
            oldBlitData.width = width;
            oldBlitData.height = height;
        }
        if (!oldBlitData.data)
        {
            HandleCu(cuMemAllocHost((void **)&oldBlitData.data, (size_t)width * height * sizeof(int) * 4));
        }
        kernel->RenderInto(oldBlitData.data, (size_t)oldBlitData.width, (size_t)oldBlitData.height);
        EnqueueSyncBlitData(kernel->Stream(), oldBlitData);
    }

    virtual void BlitFrame(BlitData pixbuf, Kernel *kernel)
    {
        EnqueueFrame(kernel); // this is above the others so that the GPU gets started as we're blitting

        SDL_Surface *surface = SDL_GetWindowSurface(window);
        Blit(pixbuf, surface);
        if (IsUserInput())
        {
            DrawUI(kernel, font, surface);
        }
        SDL_UpdateWindowSurface(window);
    }
};

struct HeadlessRenderType : public RenderType
{
    const int numFrames;
    int currentFrame;
    const int numTimes;
    int currentTime;
    BlitData mBlitData;

    HeadlessRenderType(int numFrames, int numTimes, int width, int height)
            : numFrames(numFrames), currentFrame(0), numTimes(numTimes), currentTime(0)
    {
        mBlitData.width = width;
        mBlitData.height = height;
        HandleCu(cuMemAllocHost((void **)&mBlitData.data, (size_t)width * height * sizeof(int) * 4));
    }

    ~HeadlessRenderType()
    {
        HandleCu(cuMemFreeHost(mBlitData.data));
    }

    std::string RenderstateFilename(Kernel *kernel, int currentTime)
    {
        std::ostringstream builder;
        builder << kernel->Name();
        int shiftx, shifty;
        if (RenderOffset(&shiftx, &shifty))
        {
            builder << "." << shiftx << "x" << shifty;
        }
        builder << ".renderstate";
        std::string gpuname = GpuName();
        if (!gpuname.empty())
        {
            builder << "." << gpuname;
        }
        if (numTimes > 1)
        {
            builder << ".t" << currentTime;
        }
        builder << ".clam3";
        return builder.str();
    }

    void LoadAnimationState(Kernel *kernel)
    {
        std::cout << "Loading animation keyframes\n";
        kernel->LoadAnimation();
    }

    void LoadFileState(Kernel *kernel, int time)
    {
        std::string filename = RenderstateFilename(kernel, time);
        try
        {
            // Loads *.renderstate.clam3 (the previously-left-off render)
            StateSync *sync = NewFileStateSync(filename.c_str(), true);
            kernel->RecvState(sync, true);
            delete sync;
            std::cout << "Loaded intermediate state from " << filename << "\n";
        }
        catch (const std::exception &ex)
        {
            // If that fails, try *.clam3 (the saved state parameters)
            std::cout << "Didn't load intermediate state from " << filename << ": " << ex.what()
            << "\nTrying initial headless state instead\n";
            StateSync *sync = NewFileStateSync((kernel->Name() + ".clam3").c_str(), true);
            kernel->RecvState(sync, false);
            kernel->SetFramed(true); // ensure we're not rendering with basic mode
            delete sync;
        }
    }

    void SaveProgress(Kernel *kernel, int currentTime)
    {
        std::string filename = RenderstateFilename(kernel, currentTime);
        StateSync *sync = NewFileStateSync(filename.c_str(), false);
        kernel->SendState(sync, true);
        delete sync;
        std::cout << "Saved intermediate progress to " << filename << "\n";
    }

    virtual void Boot(Kernel *kernel)
    {
        const int syncEveryNFrames = 32;
        for (int i = 0; i < syncEveryNFrames; i++)
        {
            if (EnqueueSingle(kernel, mBlitData))
            {
                break;
            }
        }
        EnqueueSyncBlitData(kernel->Stream(), mBlitData);
    }

    // returns true if current frame can't be added to anymore
    virtual bool EnqueueSingle(Kernel *kernel, BlitData blitData)
    {
        if (currentTime == 0 && currentFrame == 0)
        {
            //std::cout << "Flush/loading initial state\n";
            kernel->Resize((size_t)blitData.width, (size_t)blitData.height);
            if (numTimes > 1)
            {
                LoadAnimationState(kernel);
            }
            else
            {
                LoadFileState(kernel, currentTime);
                int frame = kernel->GetFrame();
                if (frame > 0)
                {
                    currentFrame = frame;
                }
            }
            // note we can't refactor SetFramed to here, due to possibly just loading a renderstate.clam3
        }
        if (currentFrame == 0)
        {
            if (numTimes > 1)
            {
                double time = (double)currentTime / numTimes;
                //time = -std::cos(time * 6.28318530718f) * 0.5f + 0.5f;
                kernel->SetTime(time, false); // boolean is `wrap`
            }
            kernel->SetFramed(true);
        }
        kernel->RenderInto(currentFrame == numFrames - 1 ? blitData.data : NULL,
                           (size_t)blitData.width, (size_t)blitData.height);
        currentFrame++;
        return currentFrame >= numFrames;
    }

    void WriteFrame(BlitData pixbuf, Kernel *kernel, int currentTime)
    {
        SDL_Surface *surface = SDL_CreateRGBSurface(0, pixbuf.width, pixbuf.height, 4 * 8, 255 << 16, 255 << 8, 255, 0);
        if (SDL_LockSurface(surface))
        {
            throw std::runtime_error("Could not lock temp buffer surface");
        }
        memcpy(surface->pixels, pixbuf.data, (size_t)pixbuf.width * pixbuf.height * surface->format->BytesPerPixel);
        SDL_UnlockSurface(surface);
        std::string filename = RenderstateFilename(kernel, currentTime);
        filename += ".bmp";
        SDL_SaveBMP(surface, filename.c_str());
        std::cout << "Saved image '" << filename << "'\n";
        SDL_FreeSurface(surface);
    }

    virtual void BlitFrame(BlitData pixbuf, Kernel *kernel)
    {
        if (DoSaveProgress() && numTimes <= 1)
        {
            SaveProgress(kernel, currentTime);
        }
        std::cout << numFrames - currentFrame << " frames left in this image, " <<
        numTimes - currentTime << " images left\n";
        if (currentFrame >= numFrames)
        {
            WriteFrame(pixbuf, kernel, currentTime);
            currentFrame = 0;
            currentTime++; // TODO: Multi-GPU queue
        }
        if (currentTime >= numTimes)
        {
            SDL_Event quitEvent;
            quitEvent.type = SDL_QUIT;
            SDL_PushEvent(&quitEvent);
        }
        else
        {
            Boot(kernel);
        }
    }
};

static void InitCuda(CUcontext *cuContext)
{
    if (!cudaSuccessfulInit)
    {
        throw std::runtime_error("CUDA device init failure while in compute mode");
    }
    int deviceNum = CudaDeviceNum(); // user setting
    int maxDev = 0;
    HandleCu(cuDeviceGetCount(&maxDev));
    if (maxDev <= 0)
    {
        throw std::runtime_error("cuDeviceGetCount returned zero, is an NVIDIA GPU present on the system?");
    }
    if (deviceNum < 0 || deviceNum >= maxDev)
    {
        throw std::runtime_error("Invalid device number " + tostring(deviceNum) + ": must be less than "
                                 + tostring(maxDev));
    }
    CUdevice cuDevice;
    HandleCu(cuDeviceGet(&cuDevice, deviceNum));
    {
        char name[128];
        HandleCu(cuDeviceGetName(name, sizeof(name) - 1, cuDevice));
        std::cout << "Using device (" << deviceNum << " of " << maxDev << "): " << name << "\n";
    }
    HandleCu(cuCtxCreate(cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));
    HandleCu(cuCtxSetCurrent(*cuContext));
}

Driver::Driver() : cuContext(0), connection()
{
    bool isCompute = IsCompute();
    int numHeadlessTimes;
    int headless = Headless(&numHeadlessTimes);

    SDL_Rect windowPos = GetWindowPos();

    if (isCompute)
    {
        InitCuda(&cuContext);
    }

    kernel = new Kernel(KernelName());
    if (isCompute)
    {
        if (headless > 0)
        {
            renderType = new HeadlessRenderType(headless, numHeadlessTimes, windowPos.w, windowPos.h);
        }
        else
        {
            renderType = new CpuRenderType(windowPos);
        }
    }
    else
    {
        renderType = new NoRenderType(windowPos);
    }
}

Driver::~Driver()
{
    if (renderType)
    {
        delete renderType;
    }
    if (kernel)
    {
        delete kernel;
    }
    if (cuContext)
    {
        HandleCu(cuCtxDestroy(cuContext));
    }
}

void Driver::Tick()
{
    Uint32 currentTime = SDL_GetTicks();
    double time = (currentTime - lastTickTime) / 1000.0;
    lastTickTime = currentTime;

    /*
    double weight = 1 / (fpsAverage + 1);
    fpsAverage = (timePassed + fpsAverage * weight) / (weight + 1);
    timeSinceLastTitle += timePassed;
    while (timeSinceLastTitle > 1.0)
    {
        SDL_Delay(1);
        SDL_SetWindowTitle(window, ("Clam3 - " + tostring(1 / fpsAverage) + " fps").c_str());
        timeSinceLastTitle--;
    }
    */

    if (IsUserInput())
    {
        kernel->Integrate(time);
    }
    if (connection.Sync(kernel))
    {
        SDL_Event event;
        event.type = SDL_QUIT;
        SDL_PushEvent(&event);
    }
}

void Driver::MainLoop()
{
    SDL_Event event;
    renderType->Boot(kernel);
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
            void (*func)(Driver *, void *) = (void (*)(Driver *, void *))event.user.data1;
            void *param = event.user.data2;
            func(this, param);
        }
    }
}

void EnqueueSdlEvent(void (*funcPtr)(Driver *, void *), void *data)
{
    SDL_Event event;
    event.type = SDL_USEREVENT;
    event.user.code = 0;
    event.user.data1 = (void *)funcPtr;
    event.user.data2 = data;
    SDL_PushEvent(&event);
}

static void EnqueueCuMemFreeHelper(Driver *, void *hostPtr)
{
    HandleCu(cuMemFreeHost(hostPtr));
}

void EnqueueCuMemFreeHost(void *hostPtr)
{
    EnqueueSdlEvent(EnqueueCuMemFreeHelper, hostPtr);
}

void Driver::BlitImmediate(BlitData blitData)
{
    Tick(); // well, where else am I going to put it?
    renderType->BlitFrame(blitData, kernel);
}

static void EnqueueBlitDataHelper(Driver *driver, void *blitData)
{
    BlitData *data = (BlitData *)blitData;
    BlitData temp = *data;
    delete data;
    driver->BlitImmediate(temp);
}

void EnqueueBlitData(BlitData blitData)
{
    EnqueueSdlEvent(EnqueueBlitDataHelper, new BlitData(blitData));
}
