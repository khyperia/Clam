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

    virtual bool Update(Kernel *kernel) = 0;
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

    virtual bool Update(Kernel *)
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

    void HandleGl()
    {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR)
        {
            throw std::runtime_error(std::string("OpenGL error: ") + (const char *)glGetString(err));
        }
    }

    virtual void Resize()
    {
        glViewport(0, 0, width, height);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_COPY);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
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

    virtual bool Update(Kernel *)
    {
        HandleCu(cuCtxSynchronize());

        glBindTexture(GL_TEXTURE_2D, textureID);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

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

        SDL_GL_SwapWindow(window);

        HandleGl();
        return true;
    }
};

struct HeadlessRenderType : public RenderType
{
    CuMem<int> renderBuffer;
    int numFrames;

    HeadlessRenderType(Kernel *kernel, int numFrames, bool *loadedIntermediate)
            : numFrames(numFrames)
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
        renderBuffer = CuMem<int>((size_t)width * height);
    }

    CuMem<int> &GetBuffer()
    {
        return renderBuffer;
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
            renderBuffer.CopyTo((int *)surface->pixels);
            SDL_UnlockSurface(surface);
            SDL_SaveBMP(surface, "image.bmp");
            std::cout << "Saved image 'image.bmp'" << std::endl;
            SDL_FreeSurface(surface);
            return false;
        }
        return true;
    }
};

Driver::Driver() : cuContext(0), connection(), headlessWindowSize(0, 0)
{
    bool isCompute = IsCompute();
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
        if (isGpuRenderer)
        {
            HandleCu(cuGLCtxCreate(&cuContext, 0, cuDevice));
        }
        else
        {
            HandleCu(cuCtxCreate(&cuContext, 0, cuDevice));
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
    return renderType->Update(kernel);
}