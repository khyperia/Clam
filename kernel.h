#pragma once

#include "cumem.h"
#include "util.h"
#include <unordered_map>
#include <functional>

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
            throw std::runtime_error(
                "CPU/GPU variable size didn't match: " + tostring(sizeof(T))
                    + " vs. " + tostring(gpuVar.bytesize()));
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

    std::unordered_map<std::string, std::unique_ptr<GpuKernelVar>>
        variable_cache;

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
