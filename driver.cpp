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
    size_t width, height;

    RenderType() : width(0), height(0)
    {
    }

    virtual ~RenderType()
    {
    };

    virtual bool Update(Kernel *kernel, TTF_Font *font) = 0;
};

struct CpuRenderType : public RenderType
{
    SDL_Window *window;

    CpuRenderType(SDL_Window *window) : RenderType(), window(window)
    {
    }

    ~CpuRenderType()
    {
    }

    void Blit(Kernel *kern, SDL_Surface *surface)
    {
        if (SDL_LockSurface(surface))
        {
            throw std::runtime_error("Could not lock window surface");
        }
        if (surface->format->BytesPerPixel != 4)
        {
            throw std::runtime_error("Window surface bytes/pixel != 4");
        }
        kern->RenderInto((int *)surface->pixels, width, height);
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

    virtual bool Update(Kernel *kern, TTF_Font *font)
    {
        SDL_Surface *surface = SDL_GetWindowSurface(window);
        Blit(kern, surface);
        if (IsUserInput())
        {
            DrawUI(kern, font, surface);
        }
        SDL_UpdateWindowSurface(window);
        return true;
    };
};

struct HeadlessRenderType : public RenderType
{
    int numFrames;
    int currentFrame;
    int numTimes;
    int currentTime;

    HeadlessRenderType(int numFrames, int numTimes)
            : numFrames(numFrames), currentFrame(numFrames), numTimes(numTimes), currentTime(0)
    {
    }

    ~HeadlessRenderType()
    {
    }

    std::string RenderstateFilename(Kernel *kernel)
    {
        std::ostringstream builder;
        builder << kernel->Name();
        int shiftx, shifty;
        if (RenderOffset(&shiftx, &shifty))
        {
            builder << "." << shiftx << "x" << shifty;
        }
        builder << ".renderstate";
        std::string imagename = ImageName();
        if (!imagename.empty())
        {
            builder << "." << imagename;
        }
        builder << ".clam3";
        return builder.str();
    }

    void LoadAnimationState(Kernel *kernel)
    {
        std::cout << "Loading animation keyframes\n";
        kernel->LoadAnimation();
        kernel->SetTime(0, false);
        kernel->SetFramed(true);
    }

    void LoadFileState(Kernel *kernel)
    {
        std::string filename = RenderstateFilename(kernel);
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

    void SaveProgress(Kernel *kernel, int numLeft)
    {
        std::string filename = RenderstateFilename(kernel);
        StateSync *sync = NewFileStateSync(filename.c_str(), false);
        kernel->SendState(sync, true);
        delete sync;
        std::cout << "Saved intermediate progress to " << filename << " (" << numLeft << " frames left)\n";
    }

    void SaveFinalImage(Kernel *kernel, const std::string &subext)
    {
        SDL_Surface *surface = SDL_CreateRGBSurface(0, (int)width, (int)height, 4 * 8, 255 << 16, 255 << 8, 255, 0);
        if (SDL_LockSurface(surface))
        {
            throw std::runtime_error("Could not lock temp buffer surface");
        }
        kernel->RenderInto((int *)surface->pixels, width, height);
        SDL_UnlockSurface(surface);
        std::string filename = RenderstateFilename(kernel);
        if (!subext.empty())
            filename += "." + subext;
        filename += ".bmp";
        SDL_SaveBMP(surface, filename.c_str());
        std::cout << "Saved image '" << filename << "'\n";
        SDL_FreeSurface(surface);
    }

    virtual bool Update(Kernel *kernel, TTF_Font *)
    {
        if (currentTime == 0 && currentFrame == numFrames)
        {
            std::cout << "Flush/loading initial state\n";
            // TODO: Figure out why this is somtimes needed
            kernel->RenderInto(NULL, width, height);
            if (numTimes > 1)
            {
                LoadAnimationState(kernel);
            }
            else
            {
                LoadFileState(kernel);
            }
            // note we can't refactor SetFramed to here, due to possibly just loading a renderstate.clam3
        }
        currentFrame--;
        kernel->RenderInto(NULL, width, height);

        const int saveInterval = 5;
        if (currentFrame % (1 << saveInterval) == 0 && numTimes <= 1)
        {
            if (DoSaveProgress())
            {
                SaveProgress(kernel, currentFrame);
            }
            else
            {
                kernel->Synchronize();
                std::cout << currentTime << " frames left\n";
            }
        }
        if (currentFrame == 0)
        {
            SaveFinalImage(kernel, numTimes <= 1 ? std::string("") : tostring(currentTime));
            currentTime++;
            if (currentTime >= numTimes)
            {
                return false;
            }
            double time = (double)currentTime / numTimes;
            //time = -std::cos(time * 6.28318530718f) * 0.5f + 0.5f;
            kernel->SetTime(time, false); // boolean is `wrap`
            kernel->SetFramed(true);
            currentFrame = numFrames;
        }
        return true;
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

Driver::Driver() : cuContext(0), connection(), headlessWindowSize(0, 0)
{
    bool isCompute = IsCompute();
    int numHeadlessTimes;
    int headless = Headless(&numHeadlessTimes);

    SDL_Rect windowPos = GetWindowPos();
    if (headless <= 0)
    {
        window = new DisplayWindow(windowPos.x, windowPos.y, windowPos.w, windowPos.h);
    }
    else
    {
        window = NULL;
        headlessWindowSize = Vector2<int>(windowPos.w, windowPos.h);
    }

    if (isCompute)
    {
        InitCuda(&cuContext);
    }

    kernel = new Kernel(KernelName());
    if (isCompute)
    {
        if (headless <= 0)
        {
            renderType = new CpuRenderType(window->window);
        }
        else
        {
            renderType = new HeadlessRenderType(headless, numHeadlessTimes);
        }
    }
    else
    {
        renderType = NULL;
    }
}

Driver::~Driver()
{
    if (window)
    {
        delete window;
    }
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

void Driver::UpdateWindowSize()
{
    int newWidth, newHeight;
    if (window)
    {
        SDL_GetWindowSize(window->window, &newWidth, &newHeight);
    }
    else
    {
        newWidth = headlessWindowSize.x;
        newHeight = headlessWindowSize.y;
    }
    if (newWidth <= 0 || newHeight <= 0)
    {
        throw std::runtime_error("Window size 0x0 is invalid");
    }
    if (renderType)
    {
        if ((size_t)newWidth != renderType->width || (size_t)newHeight != renderType->height)
        {
            renderType->width = (size_t)newWidth;
            renderType->height = (size_t)newHeight;
        }
    }
}

bool Driver::RunFrame(double timePassed)
{
    if (window)
    {
        if (!window->UserInput(kernel, timePassed))
        {
            return false;
        }
    }
    if (connection.Sync(kernel))
    {
        return false;
    }
    UpdateWindowSize();
    if (renderType)
    {
        bool cont = renderType->Update(kernel, window ? window->font : NULL);
        return cont;
    }
    else
    {
        kernel->UpdateNoRender();
        if (IsUserInput())
        {
            SDL_Surface *conf = kernel->Configure(window->font);
            if (conf)
            {
                SDL_Surface *surface = SDL_GetWindowSurface(window->window);
                SDL_FillRect(surface, NULL, 0);
                SDL_BlitSurface(conf, NULL, surface, NULL);
                SDL_FreeSurface(conf);
                SDL_UpdateWindowSurface(window->window);
            }
        }
        return true;
    }
}
