#include "kernel.h"
#include <fstream>

/*
class KernelModuleBase
{
    const CudaContext myContext;

protected:
    virtual void Resize(int w, int h, const CudaContext context) = 0;

    KernelModuleBase(const CudaContext context)
        : myContext(context)
    {
    }

    void CheckContext(const CudaContext context) const
    {
        if (myContext.IsValid() && context.IsValid())
        {
            if (myContext.Index() != context.Index())
            {
                throw std::runtime_error("Invalid context on KernelModule: "
                                             + tostring(context.Index())
                                             + " while this module's context is "
                                             + tostring(myContext.Index()));
            }
        }
        else if (myContext.IsValid() != context.IsValid())
        {
            throw std::runtime_error(
                "Invalid context on KernelModule: one was valid, the other wasn't");
        }
    }

public:
    virtual ~KernelModuleBase()
    {
    }

    virtual bool Update(int w,
                        int h,
                        bool resized,
                        CUstream stream,
                        const CudaContext context) = 0;

    virtual bool OneTimeKeypress(SDL_Keycode keycode) = 0;

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time) = 0;

    virtual void SendState(const StateSync *output,
                           bool everything,
                           int width,
                           int height,
                           CUstream stream,
                           const CudaContext context) const = 0;

    virtual bool RecvState(const StateSync *input,
                           bool everything,
                           int width,
                           int height,
                           CUstream stream,
                           const CudaContext context) = 0;

    virtual SDL_Surface *Configure(TTF_Font *font) const = 0;
};

template<typename T>
class KernelModule: public KernelModuleBase
{
private:
    T oldCpuVar;
protected:
    T value;
    CUdeviceptr gpuVar;
    CUmodule module;

    KernelModule(CUmodule module,
                 const char *varname,
                 const CudaContext context)
        : KernelModuleBase(context), oldCpuVar(), value(), module(module)
    {
        if (module != NULL)
        {
            size_t gpuVarSize;
            context
                .Run(cuModuleGetGlobal(&gpuVar, &gpuVarSize, module, varname));
            if (sizeof(T) != gpuVarSize)
            {
                throw std::runtime_error(
                    std::string(varname) + " size did not match actual size "
                        + tostring(gpuVarSize));
            }
        }
    }

    virtual void Update(const CudaContext context) = 0;

public:
    virtual ~KernelModule()
    {
    }

    virtual bool Update(int w,
                        int h,
                        bool changed,
                        CUstream stream,
                        const CudaContext context)
    {
        CheckContext(context);
        if (changed)
        {
            Resize(w, h, context);
        }
        Update(context);
        if (context.IsValid() && memcmp(&oldCpuVar, &value, sizeof(T)))
        {
            context.Run(cuMemcpyHtoDAsync(gpuVar, &value, sizeof(T), stream));
            oldCpuVar = value;
        }
        return changed;
    }
};

template<typename T>
class KernelSettingsModule: public KernelModule<T>
{
    SettingModule<T> *setting;

public:
    KernelSettingsModule(CUmodule module,
                         SettingModule<T> *setting,
                         const CudaContext context)
        : KernelModule<T>(module, setting->VarName(), context), setting(setting)
    {
    }

    ~KernelSettingsModule()
    {
    }

    virtual void Resize(int, int, const CudaContext)
    {
    }

    virtual void Update(const CudaContext context)
    {
        (void)context;
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

    virtual void SendState(const StateSync *output,
                           bool,
                           int,
                           int,
                           CUstream,
                           const CudaContext) const
    {
        setting->SendState(output);
    }

    virtual bool RecvState(const StateSync *input,
                           bool,
                           int,
                           int,
                           CUstream,
                           const CudaContext)
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
    const CudaContext context = CudaContext::Invalid;
    if (keycode == SDLK_t)
    {
        std::cout << "Saving state" << std::endl;
        StateSync *sync = NewFileStateSync((Name() + ".clam3").c_str(), false);
        SendState(sync, false, context);
        delete sync;
    }
    else if (keycode == SDLK_y)
    {
        std::cout << "Loading state" << std::endl;
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
                StateSync *sync =
                    NewFileStateSync((Name() + ".animation.clam3").c_str(),
                                     true);
                animation = new SettingAnimation(sync, settings[0]);
                delete sync;
                std::cout << "Loaded previous animation" << std::endl;
            }
            catch (const std::exception &e)
            {
                animation = new SettingAnimation(NULL, settings[0]);
                std::cout << "Created new animation" << std::endl;
            }
        }
        animation->AddKeyframe(settings[0]);
        StateSync *sync =
            NewFileStateSync((Name() + ".animation.clam3").c_str(), false);
        animation->WriteKeyframes(sync);
        delete sync;
        std::cout << "Added keyframe" << std::endl;
    }
    else if (keycode == SDLK_b)
    {
        if (!animation)
        {
            animation = new SettingAnimation(NULL, settings[0]);
        }
        animation->ClearKeyframes();
        StateSync *sync =
            NewFileStateSync((Name() + ".animation.clam3").c_str(), false);
        animation->WriteKeyframes(sync);
        delete sync;
        std::cout << "Cleared keyframes" << std::endl;
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

class ModuleBuffer: public KernelModule<CUdeviceptr>
{
    int elementSize;

public:
    ModuleBuffer(CUmodule module,
                 const char *varname,
                 int elementSize,
                 const CudaContext context)
        : KernelModule<CUdeviceptr>(module, varname, context),
          elementSize(elementSize)
    {
    }

    ~ModuleBuffer()
    {
        if (value && cuMemFree(value) != CUDA_SUCCESS)
        {
            std::cout << "Could not free CUDA memory"
                      << std::endl; // TODO: Wrong context?
        }
    }

    virtual void Update(const CudaContext context)
    {
        (void)context;
    }

    virtual void Resize(int width, int height, const CudaContext context)
    {
        CheckContext(context);
        if (value)
        {
            context.Run(cuMemFree(value));
        }
        context.Run(cuMemAlloc(&value, (size_t)elementSize * width * height));
    }

    virtual bool OneTimeKeypress(SDL_Keycode)
    {
        return false;
    }

    virtual bool RepeatKeypress(SDL_Keycode, double)
    {
        return false;
    }

    virtual void SendState(const StateSync *output,
                           bool everything,
                           int width,
                           int height,
                           CUstream stream,
                           const CudaContext context) const
    {
        if (everything)
        {
            CheckContext(context);
            size_t size = (size_t)elementSize * width * height;
            output->Send(size);
            char *host;
            context.Run(cuMemAllocHost((void **)&host, size));
            CuMem<char>(value, size).CopyTo(host, stream, context);
            output->SendFrom(host, size);
            context.Run(cuStreamSynchronize(stream)); // TODO: SendState sync
            context.Run(cuMemFreeHost(host));
        }
    }

    virtual bool RecvState(const StateSync *input,
                           bool everything,
                           int width,
                           int height,
                           CUstream stream,
                           const CudaContext context)
    {
        if (everything)
        {
            size_t size = input->Recv<size_t>();
            size_t existingSize = (size_t)elementSize * width * height;
            if (existingSize != size)
            {
                std::cout
                    << "Not uploading state buffer due to mismatched sizes: current "
                    << existingSize << "(" << elementSize << " * " << width
                    << " * " << height << "), new " << size << std::endl;
                char *tmp = new char[size];
                input->RecvInto(tmp, size);
                delete[] tmp;
                return false;
            }
            else
            {
                CheckContext(context);
                char *host;
                context.Run(cuMemAllocHost((void **)&host, size));
                input->RecvInto(host, size);
                CuMem<char>(value, size).CopyFrom(host, stream, context);
                context
                    .Run(cuStreamSynchronize(stream)); // TODO: RecvState sync
                context.Run(cuMemFreeHost(host));
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

extern "C" {
extern const unsigned char mandelbox[];
extern const unsigned int mandelbox_len;

extern const unsigned char mandelbrot[];
extern const unsigned int mandelbrot_len;
}

static void AddMandelbox(CUmodule cuModule,
                         std::vector<std::vector<SettingModuleBase *> > &settingAdd,
                         std::vector<std::vector<KernelModuleBase *> > &moduleAdd,
                         const CudaContext context)
{
    std::vector<SettingModuleBase *> settings;
    std::vector<KernelModuleBase *> modules;
    Module3dCameraSettings *camera = new Module3dCameraSettings();
    ModuleMandelboxSettings *mbox = new ModuleMandelboxSettings();
    settings.push_back(camera);
    settings.push_back(mbox);
    modules.push_back(new KernelSettingsModule<GpuCameraSettings>(cuModule,
                                                                  camera,
                                                                  context));
    modules.push_back(new KernelSettingsModule<MandelboxCfg>(cuModule,
                                                             mbox,
                                                             context));
    modules.push_back(new ModuleBuffer(cuModule,
                                       "BufferScratchArr",
                                       MandelboxStateSize,
                                       context));
    modules.push_back(new ModuleBuffer(cuModule,
                                       "BufferRandArr",
                                       sizeof(int) * 2,
                                       context));
    settingAdd.push_back(settings);
    moduleAdd.push_back(modules);
}

static void AddMandelbrot(CUmodule cuModule,
                          std::vector<std::vector<SettingModuleBase *> > &settingAdd,
                          std::vector<std::vector<KernelModuleBase *> > &moduleAdd,
                          const CudaContext context)
{
    std::vector<SettingModuleBase *> settings;
    std::vector<KernelModuleBase *> modules;
    Module2dCameraSettings *camera = new Module2dCameraSettings();
    ModuleJuliaBrotSettings *julia = new ModuleJuliaBrotSettings();
    settings.push_back(camera);
    settings.push_back(julia);
    modules.push_back(new KernelSettingsModule<Gpu2dCameraSettings>(cuModule,
                                                                    camera,
                                                                    context));
    modules.push_back(new KernelSettingsModule<JuliaBrotSettings>(cuModule,
                                                                  julia,
                                                                  context));
    settingAdd.push_back(settings);
    moduleAdd.push_back(modules);
}

Kernel::Kernel(std::string name, const std::vector<CudaContext> contexts)
    : name(name), contexts(contexts),
      useRenderOffset(RenderOffset(&renderOffset.x, &renderOffset.y)), frames(),
      maxLocalSize(16), animation(NULL), gpuBuffers(contexts.size()),
      oldWidth(), oldHeight()
{
    if (name.empty())
    {
        name = std::string("mandelbox");
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
            contexts[i].SetCurrent();
            CUmodule cuModule;
            contexts[i].Run(cuModuleLoadData(&cuModule,
                                             std::string((const char *)mandelbox,
                                                         mandelbox_len)
                                                 .c_str()));
            cuModules.push_back(cuModule);

            AddMandelbox(cuModule, settings, modules, contexts[i]);
        }
        if (contexts.size() == 0)
        {
            AddMandelbox(NULL, settings, modules, CudaContext::Invalid);
        }
    }
    else if (name == "mandelbrot")
    {
        for (size_t i = 0; i < contexts.size(); i++)
        {
            contexts[i].SetCurrent();
            CUmodule cuModule;
            contexts[i].Run(cuModuleLoadData(&cuModule,
                                             std::string((const char *)mandelbrot,
                                                         mandelbrot_len)
                                                 .c_str()));
            cuModules.push_back(cuModule);
            AddMandelbrot(cuModule, settings, modules, contexts[i]);
        }
        if (contexts.size() == 0)
        {
            AddMandelbrot(NULL, settings, modules, CudaContext::Invalid);
        }
    }
    else
    {
        throw std::runtime_error("Unknown kernel name " + name);
    }
    for (size_t i = 0; i < contexts.size(); i++)
    {
        contexts[i].SetCurrent();

        CUfunction kernelMain;
        contexts[i].Run(cuModuleGetFunction(&kernelMain, cuModules[i], "kern"));
        kernelMains.push_back(kernelMain);

        CUstream stream;
        contexts[i].Run(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        streams.push_back(stream);

        frames.push_back(-1);
        oldWidth.push_back(0);
        oldHeight.push_back(0);
    }
    if (frames.size() == 0)
    {
        frames.push_back(-1);
    }
}

Kernel::~Kernel()
{
    for (size_t context = 0; context < cuModules.size(); context++)
    {
        if (cuModuleUnload(cuModules[context]) != CUDA_SUCCESS)
        {
            std::cout << "Failed to unload kernel module" << std::endl;
        }
        for (size_t i = 0; i < modules[context].size(); i++)
        {
            delete modules[context][i];
        }
        for (size_t i = 0; i < settings[context].size(); i++)
        {
            delete settings[context][i];
        }
        if (cuStreamDestroy(streams[context]) != CUDA_SUCCESS)
        {
            std::cout << "Failed to destroy stream" << std::endl;
        }
    }
    if (animation)
    {
        delete animation;
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
        for (size_t i = 0; i < modules[0].size(); i++)
        {
            if (modules[0][i]->OneTimeKeypress(event.key.keysym.sym))
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
         iter != pressedKeys.end(); iter++)
    {
        for (size_t i = 0; i < modules[0].size(); i++)
        {
            if (modules[0][i]->RepeatKeypress(*iter, time))
            {
                ResetFrames(frames);
            }
        }
    }
}

void Kernel::SendState(const StateSync *output,
                       bool everything,
                       const CudaContext context) const
{
    if (everything)
    {
        if (!context.IsValid())
        {
            throw std::runtime_error(
                "Cannot send an 'everything' state with no GPU context");
        }
        output->Send(frames[context.Index()]);
    }
    else
    {
        if (!context.IsValid())
        {
            output->Send(AllNegOne(frames) ? -1 : 0);
        }
        else
        {
            output->Send(frames[context.Index()]);
        }
    }
    int index = context.IsValid() ? context.Index() : 0;
    for (size_t i = 0; i < modules[index].size(); i++)
    {
        int oldw = oldWidth.size() == 0 ? -1 : (int)oldWidth[index];
        int oldh = oldHeight.size() == 0 ? -1 : (int)oldHeight[index];
        CUstream stream = streams.size() == 0 ? NULL : streams[index];
        modules[index][i]
            ->SendState(output, everything, oldw, oldh, stream, context);
    }
}

void Kernel::RecvState(const StateSync *input,
                       bool everything,
                       const CudaContext context)
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
            if (!context.IsValid())
            {
                for (size_t i = 0; i < frames.size(); i++)
                {
                    frames[i] = -1;
                }
            }
            else
            {
                frames[context.Index()] = -1;
            }
        }
        else if (!context.IsValid())
        {
            for (size_t i = 0; i < frames.size(); i++)
            {
                if (frames[i] == -1)
                {
                    frames[i] = 0;
                }
            }
        }
        else if (context.IsValid() && frames[context.Index()] == -1)
        {
            frames[context.Index()] = 0;
        }
    }
    int index = context.IsValid() ? context.Index() : 0;
    for (size_t i = 0; i < modules[index].size(); i++)
    {
        int oldw = oldWidth.size() == 0 ? -1 : (int)oldWidth[index];
        int oldh = oldHeight.size() == 0 ? -1 : (int)oldHeight[index];
        CUstream stream = streams.size() == 0 ? NULL : streams[index];
        if (modules[index][i]
            ->RecvState(input, everything, oldw, oldh, stream, context))
        {
            if (everything)
            {
                if (!context.IsValid())
                {
                    for (size_t j = 0; j < frames.size(); j++)
                    {
                        frames[j] = loadedFrame;
                    }
                }
                else
                {
                    frames[context.Index()] = loadedFrame;
                }
            }
            else
            {
                if (context.IsValid())
                {
                    if (frames[context.Index()] != -1)
                    {
                        frames[context.Index()] = 0;
                    }
                }
                else
                {
                    ResetFrames(frames);
                }
            }
        }
    }
}

SDL_Surface *Kernel::Configure(TTF_Font *font)
{
    std::vector<SDL_Surface *> surfs;
    for (size_t i = 0; i < modules[0].size(); i++)
    {
        SDL_Surface *surf = modules[0][i]->Configure(font);
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
    SDL_Surface *wholeSurf = SDL_CreateRGBSurface(0,
                                                  width,
                                                  height,
                                                  surfs[0]->format
                                                      ->BitsPerPixel,
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
    StateSync
        *sync = NewFileStateSync((Name() + ".animation.clam3").c_str(), true);
    animation = new SettingAnimation(sync, settings[0]);
    delete sync;
}

void Kernel::SetTime(double time, bool wrap, const CudaContext context)
{
    if (!animation)
    {
        std::cout << "No animation keyframes loaded" << std::endl;
        return;
    }
    animation->Animate(settings[context.Index()], time, wrap);
}

void Kernel::SetFramed(bool framed, const CudaContext context)
{
    frames[context.Index()] = framed ? 0 : -1;
}

int Kernel::GetFrame(const CudaContext context)
{
    return frames[context.Index()];
}

void Kernel::UpdateNoRender()
{
    for (size_t i = 0; i < modules[0].size(); i++)
    {
        if (modules[0][i]->Update(-1, -1, false, NULL, CudaContext::Invalid))
        {
            ResetFrames(frames);
        }
    }
}

void Kernel::Resize(size_t width, size_t height, const CudaContext context)
{
    bool resized = false;
    if (width != oldWidth[context.Index()]
        || height != oldHeight[context.Index()])
    {
        if (oldWidth[context.Index()] != 0 || oldHeight[context.Index()] != 0)
        {
            std::cout << "Resized from " << oldWidth[context.Index()] << "x"
                      << oldHeight[context.Index()] << " to " << width << "x"
                      << height << std::endl;
        }
        oldWidth[context.Index()] = width;
        oldHeight[context.Index()] = height;
        gpuBuffers[context.Index()].Realloc(width * height, context);
        resized = true;
    }
    for (size_t i = 0; i < modules[context.Index()].size(); i++)
    {
        if (modules[context.Index()][i]->Update((int)width,
                                                (int)height,
                                                resized,
                                                streams[context.Index()],
                                                context))
        {
            ResetFrames(frames);
        }
    }
}

// NOTE: Async copy into memory param, needs a kernel->Stream() synch to finish.
void Kernel::RenderInto(int *memory,
                        size_t width,
                        size_t height,
                        const CudaContext context)
{
    Resize(width, height, context);
    int renderOffsetX = useRenderOffset ? renderOffset.x : -(int)width / 2;
    int renderOffsetY = useRenderOffset ? renderOffset.y : -(int)height / 2;
    int mywidth = (int)width;
    int myheight = (int)height;
    int myFrame = frames[context.Index()];
    if (frames[context.Index()] != -1)
    {
        frames[context.Index()]++;
    }
    void *args[] =
        {&gpuBuffers[context.Index()](), &renderOffsetX, &renderOffsetY,
         &mywidth, &myheight, &myFrame};
    unsigned int blockX = (unsigned int)maxLocalSize;
    unsigned int blockY = (unsigned int)maxLocalSize;
    unsigned int gridX = (unsigned int)(width + blockX - 1) / blockX;
    unsigned int gridY = (unsigned int)(height + blockY - 1) / blockY;
    context.Run(cuLaunchKernel(kernelMains[context.Index()],
                               gridX,
                               gridY,
                               1,
                               blockX,
                               blockY,
                               1,
                               0,
                               streams[context.Index()],
                               args,
                               NULL));
    if (memory)
    {
        gpuBuffers[context.Index()]
            .CopyTo(memory, streams[context.Index()], context);
    }
}
*/

GpuKernelVar::GpuKernelVar(const GpuKernel &kernel, const char *name)
    : gpuVar([&]() -> CuMem<char>
             {
                 CUdeviceptr deviceVar;
                 size_t deviceSize;
                 kernel.context.Run(cuModuleGetGlobal(&deviceVar,
                                                      &deviceSize,
                                                      kernel.module,
                                                      name));
                 return CuMem<char>(deviceVar, deviceSize);
             }())
{
    kernel.context.Run(cuMemAllocHost((void **)&cpuCopy, gpuVar.bytesize()));
}

GpuKernelVar::~GpuKernelVar()
{
    (void)cuMemFreeHost(cpuCopy);
}

void GpuKernelVar::Sync(const CudaContext &context, CUstream stream)
{
    if (changed)
    {
        gpuVar.CopyFrom(cpuCopy, stream, context);
        changed = false;
    }
}

void GpuKernelVar::SetData(const char *data)
{
    if (memcmp(cpuCopy, data, gpuVar.bytesize()))
    {
        memcpy(cpuCopy, data, gpuVar.bytesize());
        changed = true;
    }
}

GpuKernel::GpuKernel(CudaContext context,
                     std::function<void(int *, size_t, size_t)> render_callback,
                     const unsigned char *data,
                     size_t length)
    : context(std::move(context)), render_callback(render_callback),
      old_width(0), old_height(0)
{
    std::string data_str((const char *)data, length);
    context.Run(cuModuleLoadData(&module, data_str.c_str()));
    context.Run(cuModuleGetFunction(&main, module, "kern"));
    context.Run(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
}

GpuKernel::~GpuKernel()
{
    context.Run(cuStreamSynchronize(stream));
    if (pinned != NULL)
    {
        context.Run(cuMemFreeHost(pinned));
        pinned = NULL;
    }
    if (stream != NULL)
    {
        context.Run(cuStreamDestroy(stream));
        stream = NULL;
    }
    main = NULL;
    if (module != NULL)
    {
        context.Run(cuModuleUnload(module));
        module = NULL;
    }
}

void GpuKernel::Resize(size_t width, size_t height)
{
    if (width == old_width && height == old_height && pinned != NULL)
    {
        return;
    }
    context.Run(cuStreamSynchronize(stream));
    old_width = width;
    old_height = height;
    if (pinned != NULL)
    {
        context.Run(cuMemFreeHost(pinned));
        pinned = NULL;
    }
    size_t bytesize = width * height * sizeof(int);
    context.Run(cuMemAllocHost((void **)&pinned, bytesize));
    gpu_mem.~CuMem();
    new (&gpu_mem) CuMem<int>(context, width * height);
}

void GpuKernel::SyncVars()
{
    for (const auto &variable : variable_cache)
    {
        std::get<1>(variable)->Sync(context, stream);
    }
}

GpuKernelVar &GpuKernel::Variable(const std::string &name)
{
    if (variable_cache.count(name) == 0)
    {
        variable_cache.emplace(name, make_unique<GpuKernelVar>(*this, name.c_str()));
    }
    return *variable_cache.at(name);
}

struct streamCallbackData
{
    std::function<void(int *, size_t, size_t)> *render_callback;
    int *cpu_mem;
    size_t width, height;

    streamCallbackData(std::function<void(int *,
                                          size_t,
                                          size_t)> *render_callback,
                       int *cpu_mem,
                       size_t width,
                       size_t height)
        : render_callback(render_callback), cpu_mem(cpu_mem), width(width),
          height(height)
    {
    }
};

static void CUDA_CB streamCallback(CUstream, CUresult, void *userData)
{
    streamCallbackData *userDataPointer = (streamCallbackData *)userData;
    std::unique_ptr<streamCallbackData> data(userDataPointer);
    (*data->render_callback)(data->cpu_mem, data->width, data->height);
}

void GpuKernel::Run(int offsetX,
                    int offsetY,
                    int width,
                    int height,
                    int frame,
                    bool enqueueDownload)
{
    Resize((size_t)width, (size_t)height);
    SyncVars();
    const unsigned int maxLocalSize = 16;
    CUdeviceptr &gpu_ptr = gpu_mem();
    unsigned int blockX = maxLocalSize;
    unsigned int blockY = maxLocalSize;
    unsigned int gridX = ((unsigned int)width + blockX - 1) / blockX;
    unsigned int gridY = ((unsigned int)height + blockY - 1) / blockY;
    void *args[] = {&gpu_ptr, &offsetX, &offsetY, &width, &height, &frame};
    context.Run(cuLaunchKernel(main,
                               gridX,
                               gridY,
                               1,
                               blockX,
                               blockY,
                               1,
                               0,
                               stream,
                               args,
                               NULL));
    if (enqueueDownload)
    {
        gpu_mem.CopyTo(pinned, stream, context);
        std::unique_ptr<streamCallbackData> data =
            make_unique<streamCallbackData>(&render_callback,
                                            pinned,
                                            width,
                                            height);
        context.Run(cuStreamAddCallback(stream,
                                        streamCallback,
                                        data.release(),
                                        0));
    }
}

extern "C" {
extern const unsigned char mandelbox[];
extern const unsigned int mandelbox_len;

extern const unsigned char mandelbrot[];
extern const unsigned int mandelbrot_len;
}

KernelConfiguration::KernelConfiguration(const std::string &kernelName)
{
    if (kernelName == "mandelbox")
    {
        kernelData = mandelbox;
        kernelLength = mandelbox_len;
    }
    else if (kernelName == "mandelbrot")
    {
        kernelData = mandelbrot;
        kernelLength = mandelbrot_len;
    }
}

const unsigned char *KernelConfiguration::KernelData() const
{
    return kernelData;
}

unsigned int KernelConfiguration::KernelLength() const
{
    return kernelLength;
}
