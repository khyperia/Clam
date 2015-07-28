#include "driver.h"
#include "option.h"
#include <iostream>

struct RenderType
{
    cl::Buffer renderBuffer;
    int width, height;

    RenderType() : renderBuffer(), width(-1), height(-1)
    {
    }

    virtual ~RenderType()
    { };

    virtual void Resize() = 0;

    virtual bool Update(Kernel *kernel) = 0;
};

struct CpuRenderType : public RenderType
{
    cl::Context context;
    SDL_Window *window;

    CpuRenderType(cl::Context context, SDL_Window *window) : RenderType(), context(context), window(window)
    {
    }

    ~CpuRenderType()
    {
    }

    virtual void Resize()
    {
        renderBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(Uint8) * 4 * width * height);
    };

    virtual bool Update(Kernel *kernel)
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
        kernel->queue.enqueueReadBuffer(renderBuffer, 1, 0, sizeof(Uint8) * 4 * width * height, surface->pixels);
        SDL_UnlockSurface(surface);
        SDL_UpdateWindowSurface(window);
        return true;
    };
};

struct HeadlessRenderType : public RenderType
{
    cl::Context context;
    int numFrames;

    HeadlessRenderType(Kernel *kernel, int numFrames, bool *loadedIntermediate)
            : context(kernel->context),
              numFrames(numFrames)
    {
        std::string filename = RenderstateFilename(kernel);
        try
        {
            StateSync *sync = NewFileStateSync(filename.c_str(), true);
            kernel->LoadWholeState(sync);
            delete sync;
            std::cout << "Loaded intermediate state from " << filename << std::endl;
            *loadedIntermediate = true;
        }
        catch (const std::exception &ex)
        {
            std::cout << "Didn't load intermediate state from " << filename
            << ": " << ex.what() << std::endl;
            *loadedIntermediate = false;
        }
    }

    ~HeadlessRenderType()
    {
    }

    std::string RenderstateFilename(Kernel *kernel)
    {
        return kernel->Name() + ".renderstate.clam3";;
    }

    virtual void Resize()
    {
        renderBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(Uint8) * 4 * width * height);
    }

    virtual bool Update(Kernel *kernel)
    {
        numFrames--;
        std::cout << numFrames << " frames left" << std::endl;
        if (numFrames % 32 == 0)
        {
            std::string filename = RenderstateFilename(kernel);
            StateSync *sync = NewFileStateSync(filename.c_str(), false);
            kernel->SaveWholeState(sync);
            delete sync;
            std::cout << "Saved intermediate progress to " << filename << std::endl;
        }
        if (numFrames == 0)
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 4 * 8, 255 << 16, 255 << 8, 255, 0);
            if (SDL_LockSurface(surface))
            {
                throw std::runtime_error("Could not lock temp buffer surface");
            }
            kernel->queue.enqueueReadBuffer(renderBuffer, 1, 0, sizeof(Uint8) * 4 * width * height, surface->pixels);
            SDL_UnlockSurface(surface);
            SDL_SaveBMP(surface, "image.bmp");
            std::cout << "Saved image 'image.bmp'" << std::endl;
            SDL_FreeSurface(surface);
            return false;
        }
        kernel->queue.finish();
        return true;
    }
};

Driver::Driver() : connection(), headlessWindowSize(0, 0)
{
    bool isCompute = IsCompute();
    //isUserInput = IsUserInput();
    int headless = Headless();

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
    kernel = MakeKernel();
    if (isCompute)
    {
        if (headless <= 0)
        {
            renderType = new CpuRenderType(kernel->context, window->window);
        }
        else
        {
            bool loadedItermediate = false;
            renderType = new HeadlessRenderType(kernel, headless, &loadedItermediate);
            if (!loadedItermediate)
            {
                std::cout << "Loading initial headless state" << std::endl;
                StateSync *sync = NewFileStateSync((kernel->Name() + ".clam3").c_str(), true);
                kernel->RecvState(sync);
                delete sync;
            }
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
    connection.Sync(kernel);
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
    if (newWidth != renderType->width || newHeight != renderType->height)
    {
        renderType->width = newWidth;
        renderType->height = newHeight;
        renderType->Resize();
    }
    kernel->RenderInto(renderType->renderBuffer, (size_t)renderType->width, (size_t)renderType->height);
    return renderType->Update(kernel);
}