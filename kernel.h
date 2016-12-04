#pragma once

#include "network.h"
#include "cumem.h"
#include "vector.h"
#include <SDL.h>
#include <SDL_net.h>
#include <SDL_ttf.h>
#include <set>
#include <cuda.h>
#include <unordered_map>

/*
class SettingModuleBase;

class KernelModuleBase;

class SettingAnimation;

class Kernel
{
    std::string name;

    std::set<SDL_Keycode> pressedKeys;

    // byContext<allModules<setting> >
    std::vector<std::vector<SettingModuleBase *> > settings;
    std::vector<std::vector<KernelModuleBase *> > modules;

    const std::vector<CudaContext> contexts;
    std::vector<CUmodule> cuModules;
    std::vector<CUfunction> kernelMains;
    std::vector<CUstream> streams;

    Vector2<int> renderOffset;
    bool useRenderOffset;

    std::vector<int> frames;

    size_t maxLocalSize;

    void CommonOneTimeKeypress(SDL_Keycode keycode);

    SettingAnimation *animation;

    std::vector<CuMem<int> > gpuBuffers;
    std::vector<size_t> oldWidth;
    std::vector<size_t> oldHeight;

public:
    Kernel(std::string name, const std::vector<CudaContext> contexts);

    ~Kernel();

    void SendState(const StateSync *output,
                   bool everything,
                   const CudaContext context) const;

    void RecvState(const StateSync *input,
                   bool everything,
                   const CudaContext context);

    void UpdateNoRender();

    void Resize(size_t width, size_t height, const CudaContext context);

    void RenderInto(int *memory,
                    size_t width,
                    size_t height,
                    const CudaContext context);

    void LoadAnimation();

    void SetTime(double time, bool wrap, const CudaContext context);

    void SetFramed(bool framed, const CudaContext context);

    int GetFrame(const CudaContext context);

    SDL_Surface *Configure(TTF_Font *font);

    std::string Name();

    void UserInput(SDL_Event event);

    void Integrate(double time);

    const CUstream &Stream(const CudaContext context) const
    {
        return streams[context.Index()];
    }
};
*/

class GpuKernel;

class GpuKernelVar: public NoClone
{
    friend class GpuKernel;
    CuMem<char> gpuVar;
    char *cpuCopy;
    bool changed;

    void Sync(const CudaContext &context, CUstream stream);
    void SetData(const char *data);
public:
    template<typename T>
    void Set(const T &value)
    {
        if (sizeof(T) != gpuVar.bytesize())
        {
            throw std::runtime_error("GPU/CPU variable size didn't match");
        }
        SetData((const char *)&value);
    }
    GpuKernelVar(const GpuKernel &kernel, const char *name);
    ~GpuKernelVar();
};

class GpuKernel: public Immobile
{
    friend class GpuKernelVar;

    CudaContext context;
    CUmodule module;
    CUfunction main;
    CUstream stream;

    std::function<void(int *, size_t, size_t)> render_callback;

    CuMem<int> gpu_mem;
    int *pinned;
    size_t old_width, old_height;

    std::unordered_map<std::string, std::unique_ptr<GpuKernelVar>> variable_cache;

    void Resize(size_t width, size_t height);
    void SyncVars();
public:
    GpuKernel(CudaContext context,
              std::function<void(int *, size_t, size_t)> render_callback,
              const unsigned char *data,
              size_t length);
    ~GpuKernel();
    GpuKernelVar &Variable(const std::string &name);
    void Run(int offsetX,
             int offsetY,
             int width,
             int height,
             int frame,
             bool enqueueDownload);
};

class KernelConfiguration: public Immobile
{
    const unsigned char *kernelData;
    unsigned int kernelLength;

public:
    KernelConfiguration(const std::string &kernelName);
    const unsigned char *KernelData() const;
    unsigned int KernelLength() const;
};
