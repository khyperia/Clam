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
            std::cout << "cuInit failed" << std::endl;
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

    virtual bool Update(Kernel *kern, TTF_Font *font)
    {
        SDL_Surface *surface = SDL_GetWindowSurface(window);
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
        if (IsUserInput())
        {
            SDL_Surface *conf = kern->Configure(font);
            if (conf)
            {
                SDL_BlitSurface(conf, NULL, surface, NULL);
                SDL_FreeSurface(conf);
            }
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
        builder << ".renderstate.clam3";
        return builder.str();
    }

    virtual bool Update(Kernel *kernel, TTF_Font *)
    {
        if (currentTime == 0 && currentFrame == numFrames)
        {
            std::cout << "Flush/loading initial state" << std::endl;
            kernel->RenderInto(NULL, width, height);
            if (numTimes > 1)
            {
                std::cout << "Loading animation keyframes" << std::endl;
                kernel->LoadAnimation();
                kernel->SetTime(0, false);
                kernel->SetFramed(true);
            }
            else
            {
                std::string filename = RenderstateFilename(kernel);
                try
                {
                    StateSync *sync = NewFileStateSync(filename.c_str(), true);
                    kernel->RecvState(sync, true);
                    delete sync;
                    std::cout << "Loaded intermediate state from " << filename << std::endl;
                }
                catch (const std::exception &ex)
                {
                    std::cout << "Didn't load intermediate state from " << filename << ": " << ex.what() << std::endl
                    << "Trying initial headless state instead" << std::endl;
                    StateSync *sync = NewFileStateSync((kernel->Name() + ".clam3").c_str(), true);
                    kernel->RecvState(sync, false);
                    kernel->SetFramed(true);
                    delete sync;
                }
            }
        }
        currentFrame--;
        if (numTimes <= 1)
        {
            std::cout << currentFrame << " frames left" << std::endl;
        }
        kernel->RenderInto(NULL, width, height);
        if (currentFrame % 32 == 0 && numTimes <= 1)
        {
            std::string filename = RenderstateFilename(kernel);
            StateSync *sync = NewFileStateSync(filename.c_str(), false);
            kernel->SendState(sync, true);
            delete sync;
            std::cout << "Saved intermediate progress to " << filename << std::endl;
        }
        if (currentFrame == 0)
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, (int)width, (int)height, 4 * 8, 255 << 16, 255 << 8, 255, 0);
            if (SDL_LockSurface(surface))
            {
                throw std::runtime_error("Could not lock temp buffer surface");
            }
            kernel->RenderInto((int *)surface->pixels, width, height);
            SDL_UnlockSurface(surface);
            std::string filename = RenderstateFilename(kernel);
            filename += "." + tostring(currentTime) + ".bmp";
            SDL_SaveBMP(surface, filename.c_str());
            std::cout << "Saved image '" << filename << "'" << std::endl;
            SDL_FreeSurface(surface);
            currentTime++;
            if (currentTime >= numTimes)
            {
                return false;
            }
            double time = (double)currentTime / numTimes;
            //time = -std::cos(time * 6.28318530718f) * 0.5f + 0.5f;
            kernel->SetTime(time, false);
            kernel->SetFramed(true);
            currentFrame = numFrames;
        }
        return true;
    }
};

Driver::Driver() : cuContext(0), connection(), headlessWindowSize(0, 0)
{
    bool isCompute = IsCompute();
    int numHeadlessTimes;
    int headless = Headless(&numHeadlessTimes);

    std::string winpos = WindowPos();
    if (winpos.empty())
    {
        puts("Window position wasn't specified. Defaulting to 500x500+100+100");
        winpos = "500x500+100+100";
    }
    int x, y, width, height;
    if (sscanf(winpos.c_str(), "%dx%d%d%d", &width, &height, &x, &y) != 4)
    {
        throw std::runtime_error("Window position was in an incorrect format");
    }

    if (headless <= 0)
    {
        window = new DisplayWindow(x, y, width, height);
    }
    else
    {
        window = NULL;
        headlessWindowSize = Vector2<int>(width, height);
    }

    if (isCompute)
    {
        if (!cudaSuccessfulInit)
        {
            throw std::runtime_error("CUDA device init failure and in compute mode");
        }
        HandleCu(cuDeviceGet(&cuDevice, 0));
        {
            char name[128];
            HandleCu(cuDeviceGetName(name, sizeof(name) - 1, cuDevice));
            std::cout << "Using device: " << name << std::endl;
        }
        HandleCu(cuCtxCreate(&cuContext, CU_CTX_SCHED_YIELD, cuDevice));
        HandleCu(cuCtxSetCurrent(cuContext));
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

bool Driver::RunFrame()
{
    if (window)
    {
        if (!window->UserInput(kernel))
        {
            return false;
        }
    }
    if (connection.Sync(kernel))
    {
        return false;
    }
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
        bool cont = renderType->Update(kernel, window ? window->font : NULL);
        return cont;
    }
    else
    {
        kernel->UpdateNoRender();
        if (IsUserInput() && window != NULL)
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
