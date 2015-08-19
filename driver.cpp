#define GL_GLEXT_PROTOTYPES

#include "driver.h"
#include "option.h"
#include <GL/gl.h>
#include <cudaGL.h>

struct CudaInitClass
{
    CudaInitClass()
    {
        HandleCu(cuInit(0));
    }
} cudaInitInstance;

struct RenderType
{
    int width, height;

    RenderType() : width(-1), height(-1)
    {
    }

    virtual ~RenderType()
    {
    };

    virtual void Resize() = 0;

    virtual CuMem<int> &GetBuffer() = 0;

    virtual bool Update(Kernel *kernel, TTF_Font *font) = 0;
};

struct CpuRenderType : public RenderType
{
    CuMem<int> renderBuffer;
    SDL_Window *window;

    CpuRenderType(SDL_Window *window) : RenderType(), window(window)
    {
    }

    ~CpuRenderType()
    {
    }

    virtual void Resize()
    {
        renderBuffer = CuMem<int>((size_t)width * height);
    };

    CuMem<int> &GetBuffer()
    {
        return renderBuffer;
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
        renderBuffer.CopyTo((int *)surface->pixels);
        SDL_UnlockSurface(surface);
        SDL_Surface *conf = kern->Configure(font);
        SDL_BlitSurface(conf, NULL, surface, NULL);
        SDL_FreeSurface(conf);
        SDL_UpdateWindowSurface(window);
        return true;
    };
};

struct GpuRenderType : public RenderType
{
    SDL_Window *window;
    SDL_GLContext context;
    GLuint bufferID;
    GLuint textureID;
    CUgraphicsResource resourceCuda;
    CuMem<int> renderBuffer;

    GpuRenderType(SDL_Window *window, SDL_GLContext context) :
            RenderType(),
            window(window),
            context(context),
            bufferID(0),
            textureID(0),
            resourceCuda(NULL),
            renderBuffer()
    {
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glGenBuffers(1, &bufferID);
        glGenTextures(1, &textureID);

        HandleGl();
    }

    ~GpuRenderType()
    {
        if (resourceCuda)
        {
            cuGraphicsUnregisterResource(resourceCuda);
        }
        if (bufferID)
        {
            glDeleteBuffers(1, &bufferID);
        }
        if (textureID)
        {
            glDeleteTextures(1, &textureID);
        }
    }

    virtual void Resize()
    {
        glViewport(0, 0, width, height);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_COPY);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        if (resourceCuda)
        {
            cuGraphicsUnregisterResource(resourceCuda);
        }
        HandleCu(cuGraphicsGLRegisterBuffer(&resourceCuda, bufferID, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

        HandleCu(cuGraphicsMapResources(1, &resourceCuda, NULL));

        CUdeviceptr devPtr;
        size_t size;
        HandleCu(cuGraphicsResourceGetMappedPointer(&devPtr, &size, resourceCuda));
        renderBuffer = CuMem<int>(devPtr, size / sizeof(int));

        HandleCu(cuGraphicsUnmapResources(1, &resourceCuda, NULL));

        HandleGl();
    }

    virtual CuMem<int> &GetBuffer()
    {
        return renderBuffer;
    }

    virtual bool Update(Kernel *kern, TTF_Font *font)
    {
        HandleCu(cuCtxSynchronize());

        HandleGl();

        glBindTexture(GL_TEXTURE_2D, textureID);
        HandleGl();
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
        HandleGl();
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

        HandleGl();

        glBegin(GL_QUADS);
        glTexCoord2f(0, 1);
        glVertex3f(-1, -1, 0);
        glTexCoord2f(0, 0);
        glVertex3f(-1, 1, 0);
        glTexCoord2f(1, 0);
        glVertex3f(1, 1, 0);
        glTexCoord2f(1, 1);
        glVertex3f(1, -1, 0);
        glEnd();

        HandleGl();

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        SDL_Surface *text = kern->Configure(font);
        SDL_LockSurface(text);
        GLuint tex;
        glGenTextures(1, &tex);
        HandleGl();
        glBindTexture(GL_TEXTURE_2D, tex);
        HandleGl();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        HandleGl();
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, text->w, text->h, 0, GL_BGRA, GL_UNSIGNED_BYTE, text->pixels);
        HandleGl();
        SDL_UnlockSurface(text);
        SDL_FreeSurface(text);
        HandleGl();
        float maxx = (float)text->w / width * 2 - 1;
        float maxy = (float)text->h / height * 2 - 1;
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex3f(-1, 1, 0.1f);
        glTexCoord2f(1, 0);
        glVertex3f(maxx, 1, 0.1f);
        glTexCoord2f(1, 1);
        glVertex3f(maxx, -maxy, 0.1f);
        glTexCoord2f(0, 1);
        glVertex3f(-1, -maxy, 0.1f);
        glEnd();
        HandleGl();
        glDeleteTextures(1, &tex);
        HandleGl();

        SDL_GL_SwapWindow(window);

        HandleGl();
        return true;
    }
};

struct HeadlessRenderType : public RenderType
{
    CuMem<int> renderBuffer;
    int numFrames;
    int currentFrame;
    int numTimes;
    int currentTime;

    HeadlessRenderType(Kernel *kernel, int numFrames, int numTimes, bool *loadedIntermediate)
            : numFrames(numFrames), currentFrame(numFrames), numTimes(numTimes), currentTime(0)
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
        if (numTimes > 0)
        {
            kernel->SetTime(0);
        }
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

    virtual void Resize()
    {
        renderBuffer = CuMem<int>((size_t)width * height);
    }

    CuMem<int> &GetBuffer()
    {
        return renderBuffer;
    }

    virtual bool Update(Kernel *kernel, TTF_Font *)
    {
        currentFrame--;
        if (numTimes == 0)
        {
            std::cout << currentFrame << " frames left" << std::endl;
        }
        if (currentFrame % 32 == 0 && numTimes == 0)
        {
            std::string filename = RenderstateFilename(kernel);
            StateSync *sync = NewFileStateSync(filename.c_str(), false);
            kernel->SaveWholeState(sync);
            delete sync;
            std::cout << "Saved intermediate progress to " << filename << std::endl;
        }
        if (currentFrame == 0)
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 4 * 8, 255 << 16, 255 << 8, 255, 0);
            if (SDL_LockSurface(surface))
            {
                throw std::runtime_error("Could not lock temp buffer surface");
            }
            renderBuffer.CopyTo((int *)surface->pixels);
            SDL_UnlockSurface(surface);
            std::string filename = RenderstateFilename(kernel);
            filename += "." + tostring(currentTime) + ".bmp";
            SDL_SaveBMP(surface, filename.c_str());
            std::cout << "Saved image '" << filename << "'" << std::endl;
            SDL_FreeSurface(surface);
            if (currentTime == numTimes)
            {
                return false;
            }
            currentTime++;
            kernel->SetTime((float)currentTime / numTimes);
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

    bool isGpuRenderer = false;
    if (isCompute)
    {
        std::string renderTypeStr = RenderTypeVal();
        if (renderTypeStr.empty() || renderTypeStr == "gpu")
        {
            isGpuRenderer = true;
        }
        else if (renderTypeStr == "cpu")
        {
            isGpuRenderer = false;
        }
        else
        {
            throw std::runtime_error("Unknown renderType " + renderTypeStr);
        }
        HandleCu(cuDeviceGet(&cuDevice, 0));
        {
            char name[128];
            HandleCu(cuDeviceGetName(name, sizeof(name) - 1, cuDevice));
            std::cout << "Using device: " << name << std::endl;
        }
        if (headless <= 0 && isGpuRenderer)
        {
            HandleCu(cuGLCtxCreate(&cuContext, CU_CTX_SCHED_YIELD, cuDevice));
        }
        else
        {
            HandleCu(cuCtxCreate(&cuContext, CU_CTX_SCHED_YIELD, cuDevice));
        }
        HandleCu(cuCtxSetCurrent(cuContext));
    }

    kernel = MakeKernel();
    if (isCompute)
    {
        if (headless <= 0)
        {
            if (isGpuRenderer)
            {
                renderType = new GpuRenderType(window->window, window->context);
            }
            else
            {
                renderType = new CpuRenderType(window->window);
            }
        }
        else
        {
            bool loadedItermediate = false;
            renderType = new HeadlessRenderType(kernel, headless, numHeadlessTimes, &loadedItermediate);
            if (!loadedItermediate && IsUserInput())
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
    kernel->RenderInto(renderType->GetBuffer(), (size_t)renderType->width, (size_t)renderType->height);
    bool cont = renderType->Update(kernel, window ? window->font : NULL);
    return cont;
}