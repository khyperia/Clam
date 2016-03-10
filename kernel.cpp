#include "kernel.h"
#include "settingsModule.h"
#include "driver.h"
#include <fstream>

class KernelModuleBase
{
protected:
    virtual void Resize() = 0;

public:
    virtual ~KernelModuleBase()
    {
    }

    virtual bool Update(int w, int h, CUstream stream) = 0;

    virtual bool OneTimeKeypress(SDL_Keycode keycode) = 0;

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time) = 0;

    virtual void SendState(StateSync *output, bool everything, CUstream stream) const = 0;

    virtual bool RecvState(StateSync *input, bool everything, CUstream stream) = 0;

    virtual SDL_Surface *Configure(TTF_Font *font) const = 0;
};

template<typename T>
class KernelModule : public KernelModuleBase
{
private:
    T oldCpuVar;
protected:
    T value;
    CUdeviceptr gpuVar;
    CUmodule module;
    int width;
    int height;

    KernelModule(CUmodule module, const char *varname) :
            oldCpuVar(), value(), gpuVar(0), module(module), width(-1), height(-1)
    {
        if (module)
        {
            size_t gpuVarSize;
            HandleCu(cuModuleGetGlobal(&gpuVar, &gpuVarSize, module, varname));
            if (sizeof(T) != gpuVarSize)
            {
                throw std::runtime_error(
                        std::string(varname) + " size did not match actual size " + tostring(gpuVarSize));
            }
        }
    }

    virtual void Update() = 0;

public:
    virtual ~KernelModule()
    {
    }

    virtual bool Update(int w, int h, CUstream stream)
    {
        bool changed = width != w || height != h;
        if (changed)
        {
            width = w;
            height = h;
            Resize();
        }
        Update();
        if (memcmp(&oldCpuVar, &value, sizeof(T)))
        {
            if (gpuVar)
            {
                HandleCu(cuMemcpyHtoDAsync(gpuVar, &value, sizeof(T), stream));
            }
            oldCpuVar = value;
        }
        return changed;
    }
};

template<typename T>
class KernelSettingsModule : public KernelModule<T>
{
    SettingModule<T> *setting;

public:
    KernelSettingsModule(CUmodule module, SettingModule<T> *setting) :
            KernelModule<T>(module, setting->VarName()),
            setting(setting)
    {
    }

    ~KernelSettingsModule()
    {
    }

    virtual void Resize()
    {
    }

    virtual void Update()
    {
        setting->Update();
        setting->Apply(this->value);
    }

    virtual bool OneTimeKeypress(SDL_Keycode keycode)
    {
        return setting->OneTimeKeypress(keycode);
    }

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time)
    {
        return setting->RepeatKeypress(keycode, time);
    }

    virtual void SendState(StateSync *output, bool everything, CUstream) const
    {
        setting->SendState(output, everything);
    }

    virtual bool RecvState(StateSync *input, bool everything, CUstream)
    {
        return setting->RecvState(input, everything);
    }

    virtual SDL_Surface *Configure(TTF_Font *font) const
    {
        return setting->Configure(font);
    }
};

void Kernel::CommonOneTimeKeypress(SDL_Keycode keycode)
{
    if (keycode == SDLK_t)
    {
        std::cout << "Saving state\n";
        StateSync *sync = NewFileStateSync((Name() + ".clam3").c_str(), false);
        SendState(sync, false);
        delete sync;
    }
    else if (keycode == SDLK_y)
    {
        std::cout << "Loading state\n";
        StateSync *sync = NewFileStateSync((Name() + ".clam3").c_str(), true);
        RecvState(sync, false);
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
        frame = frame == -1 ? 0 : -1;
    }
}

class ModuleBuffer : public KernelModule<CUdeviceptr>
{
    int elementSize;

public:
    ModuleBuffer(CUmodule module, const char *varname, int elementSize) :
            KernelModule<CUdeviceptr>(module, varname),
            elementSize(elementSize)
    {
        value = 0;
    }

    ~ModuleBuffer()
    {
        if (value)
        {
            if (cuMemFree(value) != CUDA_SUCCESS)
            {
                std::cout << "Could not free CUDA memory\n";
            }
        }
    }

    virtual void Update()
    {
    }

    virtual void Resize()
    {
        if (value)
        {
            HandleCu(cuMemFree(value));
        }
        HandleCu(cuMemAlloc(&value, (size_t)elementSize * width * height));
    }

    virtual bool OneTimeKeypress(SDL_Keycode)
    {
        return false;
    }

    virtual bool RepeatKeypress(SDL_Keycode, double)
    {
        return false;
    }

    virtual void SendState(StateSync *output, bool everything, CUstream stream) const
    {
        if (everything)
        {
            size_t size = (size_t)elementSize * width * height;
            output->Send(size);
            char *host;
            HandleCu(cuMemAllocHost((void**)&host, size));
            CuMem<char>(value, size).CopyTo(host, stream);
            output->SendFrom(host, size);
            HandleCu(cuStreamSynchronize(stream)); // TODO: SendState sync
            HandleCu(cuMemFreeHost(host));
        }
    }

    virtual bool RecvState(StateSync *input, bool everything, CUstream stream)
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
                HandleCu(cuMemAllocHost((void**)&host, size));
                input->RecvInto(host, size);
                CuMem<char>(value, size).CopyFrom(host, stream);
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

Kernel::Kernel(std::string name) :
        name(name),
        useRenderOffset(RenderOffset(&renderOffset.x, &renderOffset.y)),
        frame(-1),
        maxLocalSize(16),
        animation(NULL),
        oldWidth(0),
        oldHeight(0)
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
        if (isCompute)
        {
            HandleCu(cuModuleLoadData(&cuModule, std::string((const char *)mandelbox, mandelbox_len).c_str()));
        }
        else
        {
            cuModule = NULL;
        }
        Module3dCameraSettings *camera = new Module3dCameraSettings();
        ModuleMandelboxSettings *mbox = new ModuleMandelboxSettings();
        settings.push_back(camera);
        settings.push_back(mbox);
        modules.push_back(new KernelSettingsModule<GpuCameraSettings>(cuModule, camera));
        modules.push_back(new KernelSettingsModule<MandelboxCfg>(cuModule, mbox));
        modules.push_back(new ModuleBuffer(cuModule, "BufferScratchArr", MandelboxStateSize));
        modules.push_back(new ModuleBuffer(cuModule, "BufferRandArr", sizeof(int) * 2));
    }
    else if (name == "mandelbrot")
    {
        if (isCompute)
        {
            HandleCu(cuModuleLoadData(&cuModule, std::string((const char *)mandelbrot, mandelbrot_len).c_str()));
        }
        else
        {
            cuModule = NULL;
        }
        Module2dCameraSettings *camera = new Module2dCameraSettings();
        ModuleJuliaBrotSettings *julia = new ModuleJuliaBrotSettings();
        settings.push_back(camera);
        settings.push_back(julia);
        modules.push_back(new KernelSettingsModule<Gpu2dCameraSettings>(cuModule, camera));
        modules.push_back(new KernelSettingsModule<JuliaBrotSettings>(cuModule, julia));
    }
    else
    {
        throw std::runtime_error("Unknown kernel name " + name);
    }
    if (cuModule)
    {
        HandleCu(cuModuleGetFunction(&kernelMain, cuModule, "kern"));
    }
    HandleCu(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
}

Kernel::~Kernel()
{
    if (cuModule)
    {
        if (cuModuleUnload(cuModule) != CUDA_SUCCESS)
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
    if (stream)
    {
        if (cuStreamDestroy(stream) != CUDA_SUCCESS)
        {
            std::cout << "Failed to destroy stream\n";
        }
    }
}

static void ResetFrame(int &frame)
{
    if (frame != -1)
    {
        frame = 0;
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
                ResetFrame(frame);
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
                ResetFrame(frame);
            }
        }
    }
}

void Kernel::SendState(StateSync *output, bool everything) const
{
    if (everything)
    {
        output->Send(frame);
    }
    else
    {
        output->Send(frame == -1 ? -1 : 0);
    }
    for (size_t i = 0; i < modules.size(); i++)
    {
        modules[i]->SendState(output, everything, stream);
    }
}

void Kernel::RecvState(StateSync *input, bool everything)
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
            frame = -1;
        }
        else if (frame == -1)
        {
            frame = 0;
        }
    }
    for (size_t i = 0; i < modules.size(); i++)
    {
        if (modules[i]->RecvState(input, everything, stream))
        {
            if (everything)
            {
                frame = loadedFrame;
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

void Kernel::SetFramed(bool framed)
{
    frame = framed ? 0 : -1;
}

int Kernel::GetFrame()
{
    return frame;
}

void Kernel::UpdateNoRender()
{
    for (size_t i = 0; i < modules.size(); i++)
    {
        if (modules[i]->Update(-1, -1, stream))
        {
            ResetFrame(frame);
        }
    }
}

void Kernel::Resize(size_t width, size_t height)
{
    if (width != oldWidth || height != oldHeight)
    {
        if (oldWidth != 0 || oldHeight != 0)
        {
            std::cout << "Resized from " << oldWidth << "x" << oldHeight <<
            " to " << width << "x" << height << "\n";
        }
        oldWidth = width;
        oldHeight = height;
        gpuBuffer.Realloc(width * height);
    }
    for (size_t i = 0; i < modules.size(); i++)
    {
        if (modules[i]->Update((int)width, (int)height, stream))
        {
            ResetFrame(frame);
        }
    }
}

// NOTE: Async copy into memory param, needs a kernel->Stream() synch to finish.
void Kernel::RenderInto(int *memory, size_t width, size_t height)
{
    Resize(width, height);
    int renderOffsetX = useRenderOffset ? renderOffset.x : -(int)width / 2;
    int renderOffsetY = useRenderOffset ? renderOffset.y : -(int)height / 2;
    int mywidth = (int)width;
    int myheight = (int)height;
    int myFrame = frame;
    if (frame != -1)
    {
        frame++;
    }
    void *args[] =
            {
                    &gpuBuffer(),
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
    HandleCu(cuLaunchKernel(kernelMain, gridX, gridY, 1, blockX, blockY, 1, 0, stream, args, NULL));
    if (memory)
    {
        gpuBuffer.CopyTo(memory, stream);
    }
}
