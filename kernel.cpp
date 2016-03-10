#include "kernel.h"
#include "settingsModule.h"
#include "driver.h"
#include <fstream>

class KernelModuleBase
{
protected:
    virtual void Resize(int w, int h, int context) = 0;

public:
    virtual ~KernelModuleBase()
    {
    }

    virtual bool Update(int w, int h, bool resized, std::vector<CUstream> streams, int context) = 0;

    virtual bool OneTimeKeypress(SDL_Keycode keycode) = 0;

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time) = 0;

    virtual void SendState(StateSync *output, bool everything, int width, int height, CUstream stream,
                           int context) const = 0;

    virtual bool RecvState(StateSync *input, bool everything, int width, int height, CUstream stream, int context) = 0;

    virtual SDL_Surface *Configure(TTF_Font *font) const = 0;
};

template<typename T>
class KernelModule : public KernelModuleBase
{
private:
    std::vector<T> oldCpuVar;
protected:
    std::vector<T> value;
    std::vector<CUcontext> contexts;
    std::vector<CUdeviceptr> gpuVars;
    std::vector<CUmodule> modules;

    KernelModule(std::vector<CUcontext> contexts, std::vector<CUmodule> modules, const char *varname) :
            oldCpuVar(), value(), contexts(contexts), gpuVars(), modules(modules)
    {
        for (size_t i = 0; i < modules.size(); i++)
        {
            size_t gpuVarSize;
            CUdeviceptr gpuVar;
            HandleCu(cuCtxSetCurrent(contexts[i]));
            HandleCu(cuModuleGetGlobal(&gpuVar, &gpuVarSize, modules[i], varname));
            gpuVars.push_back(gpuVar);
            if (sizeof(T) != gpuVarSize)
            {
                throw std::runtime_error(
                        std::string(varname) + " size did not match actual size " + tostring(gpuVarSize));
            }
            value.push_back(T());
            oldCpuVar.push_back(T());
        }
        if (value.size() == 0)
        {
            value.push_back(T());
            oldCpuVar.push_back(T());
        }
    }

    virtual void Update(int context) = 0;

public:
    virtual ~KernelModule()
    {
    }

    virtual bool Update(int w, int h, bool changed, std::vector<CUstream> streams, int context)
    {
        if (changed)
        {
            Resize(w, h, context);
        }
        Update(context);
        if (gpuVars.size() > 0 && memcmp(&oldCpuVar[context], &value[context], sizeof(T)))
        {
            HandleCu(cuMemcpyHtoDAsync(gpuVars[context], &value[context], sizeof(T), streams[context]));
            oldCpuVar[context] = value[context];
        }
        return changed;
    }
};

template<typename T>
class KernelSettingsModule : public KernelModule<T>
{
    SettingModule<T> *setting;

public:
    KernelSettingsModule(std::vector<CUcontext> contexts, std::vector<CUmodule> modules, SettingModule<T> *setting) :
            KernelModule<T>(contexts, modules, setting->VarName()),
            setting(setting)
    {
    }

    ~KernelSettingsModule()
    {
    }

    virtual void Resize(int, int, int)
    {
    }

    virtual void Update(int context)
    {
        setting->Update();
        setting->Apply(this->value[context]);
    }

    virtual bool OneTimeKeypress(SDL_Keycode keycode)
    {
        return setting->OneTimeKeypress(keycode);
    }

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time)
    {
        return setting->RepeatKeypress(keycode, time);
    }

    virtual void SendState(StateSync *output, bool, int, int, CUstream, int) const
    {
        setting->SendState(output);
    }

    virtual bool RecvState(StateSync *input, bool, int, int, CUstream, int)
    {
        return setting->RecvState(input);
    }

    virtual SDL_Surface *Configure(TTF_Font *font) const
    {
        return setting->Configure(font);
    }
};

static bool AllNegOne(const std::vector<int> &frames)
{
    for (size_t i = 0; i < frames.size(); i++)
    {
        if (frames[i] != -1)
        {
            return false;
        }
    }
    return true;
}

void Kernel::CommonOneTimeKeypress(SDL_Keycode keycode)
{
    // no context is okay because non-everything sync doesn't use GPU
    const int context = -1;
    if (keycode == SDLK_t)
    {
        std::cout << "Saving state\n";
        StateSync *sync = NewFileStateSync((Name() + ".clam3").c_str(), false);
        SendState(sync, false, context);
        delete sync;
    }
    else if (keycode == SDLK_y)
    {
        std::cout << "Loading state\n";
        StateSync *sync = NewFileStateSync((Name() + ".clam3").c_str(), true);
        RecvState(sync, false, context);
        delete sync;
    }
    else if (keycode == SDLK_v)
    {
        if (!animation)
        {
            try
            {
                StateSync *sync = NewFileStateSync((Name() + ".animation.clam3").c_str(), true);
                animation = new SettingAnimation(sync, settings);
                delete sync;
                std::cout << "Loaded previous animation\n";
            }
            catch (const std::exception &e)
            {
                animation = new SettingAnimation(NULL, settings);
                std::cout << "Created new animation\n";
            }
        }
        animation->AddKeyframe(settings);
        StateSync *sync = NewFileStateSync((Name() + ".animation.clam3").c_str(), false);
        animation->WriteKeyframes(sync);
        delete sync;
        std::cout << "Added keyframe\n";
    }
    else if (keycode == SDLK_b)
    {
        if (!animation)
        {
            animation = new SettingAnimation(NULL, settings);
        }
        animation->ClearKeyframes();
        StateSync *sync = NewFileStateSync((Name() + ".animation.clam3").c_str(), false);
        animation->WriteKeyframes(sync);
        delete sync;
        std::cout << "Cleared keyframes\n";
    }
    else if (keycode == SDLK_x)
    {
        bool allNegOne = AllNegOne(frames);
        for (size_t i = 0; i < frames.size(); i++)
        {
            frames[i] = allNegOne ? 0 : -1;
        }
    }
}

class ModuleBuffer : public KernelModule<CUdeviceptr>
{
    int elementSize;

public:
    ModuleBuffer(std::vector<CUcontext> contexts, std::vector<CUmodule> modules, const char *varname, int elementSize) :
            KernelModule<CUdeviceptr>(contexts, modules, varname),
            elementSize(elementSize)
    {
    }

    ~ModuleBuffer()
    {
        for (size_t i = 0; i < value.size(); i++)
        {
            if (value[i] && cuMemFree(value[i]) != CUDA_SUCCESS)
            {
                std::cout << "Could not free CUDA memory\n";
            }
        }
    }

    virtual void Update(int context)
    {
        (void)context;
    }

    virtual void Resize(int width, int height, int context)
    {
        (void)context; // should already be set context
        if (value[context])
        {
            HandleCu(cuMemFree(value[context]));
        }
        HandleCu(cuMemAlloc(&value[context], (size_t)elementSize * width * height));
    }

    virtual bool OneTimeKeypress(SDL_Keycode)
    {
        return false;
    }

    virtual bool RepeatKeypress(SDL_Keycode, double)
    {
        return false;
    }

    virtual void SendState(StateSync *output, bool everything, int width, int height, CUstream stream,
                           int context) const
    {
        if (everything)
        {
            size_t size = (size_t)elementSize * width * height;
            output->Send(size);
            char *host;
            HandleCu(cuMemAllocHost((void **)&host, size));
            CuMem<char>(value[context], size).CopyTo(host, stream, context);
            output->SendFrom(host, size);
            HandleCu(cuStreamSynchronize(stream)); // TODO: SendState sync
            HandleCu(cuMemFreeHost(host));
        }
    }

    virtual bool RecvState(StateSync *input, bool everything, int width, int height, CUstream stream, int context)
    {
        if (everything)
        {
            size_t size = input->Recv<size_t>();
            size_t existingSize = (size_t)elementSize * width * height;
            if (existingSize != size)
            {
                std::cout << "Not uploading state buffer due to mismatched sizes: current "
                << existingSize << "(" << elementSize << " * " << width << " * " << height
                << "), new " << size << "\n";
                char *tmp = new char[size];
                input->RecvInto(tmp, size);
                delete tmp;
                return false;
            }
            else
            {
                char *host;
                HandleCu(cuMemAllocHost((void **)&host, size));
                input->RecvInto(host, size);
                CuMem<char>(value[context], size).CopyFrom(host, stream, context);
                HandleCu(cuStreamSynchronize(stream)); // TODO: RecvState sync
                HandleCu(cuMemFreeHost(host));
                return true;
            }
        }
        return false;
    }

    virtual SDL_Surface *Configure(TTF_Font *) const
    {
        return NULL;
    }
};

extern const unsigned char mandelbox[];
extern const unsigned int mandelbox_len;

extern const unsigned char mandelbrot[];
extern const unsigned int mandelbrot_len;

Kernel::Kernel(std::string name, std::vector<CUcontext> contexts) :
        name(name),
        contexts(contexts),
        useRenderOffset(RenderOffset(&renderOffset.x, &renderOffset.y)),
        frames(),
        maxLocalSize(16),
        animation(NULL),
        gpuBuffers(contexts.size()),
        oldWidth(),
        oldHeight()
{
    if (name.empty())
    {
        name = "mandelbox";
        this->name = name;
    }
    bool isCompute = IsCompute();
    if (isCompute && !cudaSuccessfulInit)
    {
        throw std::runtime_error("CUDA device init failure and in compute mode");
    }
    if (name == "mandelbox")
    {
        for (size_t i = 0; i < contexts.size(); i++)
        {
            HandleCu(cuCtxSetCurrent(contexts[i]));
            CUmodule cuModule;
            HandleCu(cuModuleLoadData(&cuModule, std::string((const char *)mandelbox, mandelbox_len).c_str()));
            cuModules.push_back(cuModule);
        }
        Module3dCameraSettings *camera = new Module3dCameraSettings();
        ModuleMandelboxSettings *mbox = new ModuleMandelboxSettings();
        settings.push_back(camera);
        settings.push_back(mbox);
        modules.push_back(new KernelSettingsModule<GpuCameraSettings>(contexts, cuModules, camera));
        modules.push_back(new KernelSettingsModule<MandelboxCfg>(contexts, cuModules, mbox));
        modules.push_back(new ModuleBuffer(contexts, cuModules, "BufferScratchArr", MandelboxStateSize));
        modules.push_back(new ModuleBuffer(contexts, cuModules, "BufferRandArr", sizeof(int) * 2));
    }
    else if (name == "mandelbrot")
    {
        for (size_t i = 0; i < contexts.size(); i++)
        {
            HandleCu(cuCtxSetCurrent(contexts[i]));
            CUmodule cuModule;
            HandleCu(cuModuleLoadData(&cuModule, std::string((const char *)mandelbrot, mandelbrot_len).c_str()));
            cuModules.push_back(cuModule);
        }
        Module2dCameraSettings *camera = new Module2dCameraSettings();
        ModuleJuliaBrotSettings *julia = new ModuleJuliaBrotSettings();
        settings.push_back(camera);
        settings.push_back(julia);
        modules.push_back(new KernelSettingsModule<Gpu2dCameraSettings>(contexts, cuModules, camera));
        modules.push_back(new KernelSettingsModule<JuliaBrotSettings>(contexts, cuModules, julia));
    }
    else
    {
        throw std::runtime_error("Unknown kernel name " + name);
    }
    for (size_t i = 0; i < contexts.size(); i++)
    {
        HandleCu(cuCtxSetCurrent(contexts[i]));

        CUfunction kernelMain;
        HandleCu(cuModuleGetFunction(&kernelMain, cuModules[i], "kern"));
        kernelMains.push_back(kernelMain);

        CUstream stream;
        HandleCu(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        streams.push_back(stream);

        frames.push_back(-1);
        oldWidth.push_back(0);
        oldHeight.push_back(0);
    }
}

Kernel::~Kernel()
{
    for (size_t i = 0; i < cuModules.size(); i++)
    {
        if (cuModuleUnload(cuModules[i]) != CUDA_SUCCESS)
        {
            std::cout << "Failed to unload kernel module\n";
        }
    }
    for (size_t i = 0; i < modules.size(); i++)
    {
        delete modules[i];
    }
    for (size_t i = 0; i < settings.size(); i++)
    {
        delete settings[i];
    }
    if (animation)
    {
        delete animation;
    }
    for (size_t i = 0; i < streams.size(); i++)
    {
        if (cuStreamDestroy(streams[i]) != CUDA_SUCCESS)
        {
            std::cout << "Failed to destroy stream\n";
        }
    }
}

static void ResetFrames(std::vector<int> &frame)
{
    for (size_t i = 0; i < frame.size(); i++)
    {
        if (frame[i] != -1)
        {
            frame[i] = 0;
        }
    }
}

std::string Kernel::Name()
{
    return name;
}

void Kernel::UserInput(SDL_Event event)
{
    if (event.type == SDL_KEYDOWN)
    {
        for (size_t i = 0; i < modules.size(); i++)
        {
            if (modules[i]->OneTimeKeypress(event.key.keysym.sym))
            {
                ResetFrames(frames);
            }
        }
        CommonOneTimeKeypress(event.key.keysym.sym);
        pressedKeys.insert(event.key.keysym.sym);
    }
    else if (event.type == SDL_KEYUP)
    {
        pressedKeys.erase(event.key.keysym.sym);
    }
}

void Kernel::Integrate(double time)
{
    for (std::set<SDL_Keycode>::const_iterator iter = pressedKeys.begin();
         iter != pressedKeys.end();
         iter++)
    {
        for (size_t i = 0; i < modules.size(); i++)
        {
            if (modules[i]->RepeatKeypress(*iter, time))
            {
                ResetFrames(frames);
            }
        }
    }
}

void Kernel::SendState(StateSync *output, bool everything, int context) const
{
    if (everything)
    {
        output->Send(frames[context]);
    }
    else
    {
        if (context == -1)
        {
            output->Send(AllNegOne(frames) ? -1 : 0);
        }
        else
        {
            output->Send(frames[context]);
        }
    }
    for (size_t i = 0; i < modules.size(); i++)
    {
        modules[i]->SendState(output, everything, (int)oldWidth[context], (int)oldHeight[context],
                              streams[context == -1 ? 0 : context], context);
    }
}

void Kernel::RecvState(StateSync *input, bool everything, int context)
{
    int loadedFrame = 0;
    if (everything)
    {
        loadedFrame = input->Recv<int>();
    }
    else
    {
        if (input->Recv<int>() == -1)
        {
            if (context == -1)
            {
                for (size_t i = 0; i < frames.size(); i++)
                {
                    frames[i] = -1;
                }
            }
            else
            {
                frames[context] = -1;
            }
        }
        else if (context == -1 && AllNegOne(frames))
        {
            for (size_t i = 0; i < frames.size(); i++)
            {
                frames[i] = 0;
            }
        }
        else if (context != -1 && frames[context] == -1)
        {
            frames[context] = 0;
        }
    }
    for (size_t i = 0; i < modules.size(); i++)
    {
        if (modules[i]->RecvState(input, everything, (int)oldWidth[context], (int)oldHeight[context],
                                  streams[context], context))
        {
            if (everything)
            {
                if (context == -1)
                {
                    for (size_t j = 0; j < frames.size(); j++)
                    {
                        frames[j] = loadedFrame;
                    }
                }
                else
                {
                    frames[context] = loadedFrame;
                }
            }
        }
    }
}

SDL_Surface *Kernel::Configure(TTF_Font *font)
{
    std::vector<SDL_Surface *> surfs;
    for (size_t i = 0; i < modules.size(); i++)
    {
        SDL_Surface *surf = modules[i]->Configure(font);
        if (surf)
        {
            surfs.push_back(surf);
        }
    }
    if (surfs.size() == 0)
    {
        return NULL;
    }
    if (surfs.size() == 1)
    {
        return surfs[0];
    }
    int height = 0;
    int width = 0;
    for (size_t i = 0; i < surfs.size(); i++)
    {
        height += surfs[i]->h;
        if (surfs[i]->w > width)
        {
            width = surfs[i]->w;
        }
    }
    SDL_Surface *wholeSurf = SDL_CreateRGBSurface(
            0, width, height,
            surfs[0]->format->BitsPerPixel,
            surfs[0]->format->Rmask,
            surfs[0]->format->Gmask,
            surfs[0]->format->Bmask,
            surfs[0]->format->Amask);
    height = 0;
    for (size_t i = 0; i < surfs.size(); i++)
    {
        SDL_Rect dest;
        dest.x = 0;
        dest.y = height;
        dest.w = surfs[i]->w;
        dest.h = surfs[i]->h;
        SDL_BlitSurface(surfs[i], NULL, wholeSurf, &dest);
        height += surfs[i]->h;
        SDL_FreeSurface(surfs[i]);
    }
    return wholeSurf;
}

void Kernel::LoadAnimation()
{
    if (animation)
    {
        delete animation;
    }
    StateSync *sync = NewFileStateSync((Name() + ".animation.clam3").c_str(), true);
    animation = new SettingAnimation(sync, settings);
    delete sync;
}

void Kernel::SetTime(double time, bool wrap)
{
    if (!animation)
    {
        std::cout << "No animation keyframes loaded\n";
        return;
    }
    animation->Animate(settings, time, wrap);
}

void Kernel::SetFramed(bool framed, int context)
{
    frames[context] = framed ? 0 : -1;
}

int Kernel::GetFrame(int context)
{
    return frames[context];
}

void Kernel::UpdateNoRender()
{
    for (size_t i = 0; i < modules.size(); i++)
    {
        if (modules[i]->Update(-1, -1, false, std::vector<CUstream>(), -1))
        {
            ResetFrames(frames);
        }
    }
}

void Kernel::Resize(size_t width, size_t height, int context)
{
    bool resized = false;
    if (width != oldWidth[context] || height != oldHeight[context])
    {
        if (oldWidth[context] != 0 || oldHeight[context] != 0)
        {
            std::cout << "Resized from " << oldWidth[context] << "x" << oldHeight[context] <<
            " to " << width << "x" << height << "\n";
        }
        oldWidth[context] = width;
        oldHeight[context] = height;
        gpuBuffers[context].Realloc(width * height, context);
        resized = true;
    }
    for (size_t i = 0; i < modules.size(); i++)
    {
        if (modules[i]->Update((int)width, (int)height, resized, streams, context))
        {
            ResetFrames(frames);
        }
    }
}

// NOTE: Async copy into memory param, needs a kernel->Stream() synch to finish.
void Kernel::RenderInto(int *memory, size_t width, size_t height, int context)
{
    Resize(width, height, context);
    int renderOffsetX = useRenderOffset ? renderOffset.x : -(int)width / 2;
    int renderOffsetY = useRenderOffset ? renderOffset.y : -(int)height / 2;
    int mywidth = (int)width;
    int myheight = (int)height;
    int myFrame = frames[context];
    if (frames[context] != -1)
    {
        frames[context]++;
    }
    void *args[] =
            {
                    &gpuBuffers[context](),
                    &renderOffsetX,
                    &renderOffsetY,
                    &mywidth,
                    &myheight,
                    &myFrame
            };
    unsigned int blockX = (unsigned int)maxLocalSize;
    unsigned int blockY = (unsigned int)maxLocalSize;
    unsigned int gridX = (unsigned int)(width + blockX - 1) / blockX;
    unsigned int gridY = (unsigned int)(height + blockY - 1) / blockY;
    HandleCu(cuLaunchKernel(kernelMains[context], gridX, gridY, 1, blockX, blockY, 1, 0, streams[context], args, NULL));
    if (memory)
    {
        gpuBuffers[context].CopyTo(memory, streams[context], context);
    }
}
