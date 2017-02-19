#include "kernel.h"
#include <fstream>
#include <cstring>

GpuKernelVar::GpuKernelVar(const GpuKernel &kernel, const char *name) : gpuVar(
    [&]() -> CuMem<char>
    {
        CUdeviceptr deviceVar;
        size_t deviceSize;
        kernel.context.Run(
            cuModuleGetGlobal(
                &deviceVar, &deviceSize, kernel.module, name
            ));
        return CuMem<char>(
            deviceVar, deviceSize
        );
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

GpuKernel::GpuKernel(
    CudaContext context,
    std::function<void(int *, size_t, size_t)> render_callback,
    const unsigned char *data,
    size_t length
) :
    context(std::move(context)),
    render_callback(render_callback),
    pinned(NULL),
    old_width(0),
    old_height(0)
{
    std::string data_str((const char *)data, length);
    this->context.Run(cuModuleLoadData(&module, data_str.c_str()));
    this->context.Run(cuModuleGetFunction(&main, module, "kern"));
    this->context.Run(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
}

GpuKernel::~GpuKernel()
{
    SyncStream();
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

CudaContext &GpuKernel::Context()
{
    return context;
}

void GpuKernel::SyncStream()
{
    context.Run(cuStreamSynchronize(stream));
}

void GpuKernel::Resize(size_t width, size_t height)
{
    if (width == old_width && height == old_height && pinned != NULL)
    {
        return;
    }
    SyncStream();
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
    new(&gpu_mem) CuMem<int>(context, width * height);
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

    streamCallbackData(
        std::function<void(int *, size_t, size_t)> *render_callback,
        int *cpu_mem,
        size_t width,
        size_t height
    ) : render_callback(render_callback), cpu_mem(cpu_mem), width(width), height(height)
    {
    }
};

static void CUDA_CB streamCallback(CUstream, CUresult, void *userData)
{
    streamCallbackData *userDataPointer = (streamCallbackData *)userData;
    std::unique_ptr<streamCallbackData> data(userDataPointer);
    (*data->render_callback)(data->cpu_mem, data->width, data->height);
}

void GpuKernel::Run(int offsetX, int offsetY, int width, int height, int type, bool enqueueDownload)
{
    context.SetCurrent();
    Resize((size_t)width, (size_t)height);
    SyncVars();
    const unsigned int maxLocalSize = 16;
    CUdeviceptr &gpu_ptr = gpu_mem();
    unsigned int blockX = maxLocalSize;
    unsigned int blockY = maxLocalSize;
    unsigned int gridX = ((unsigned int)width + blockX - 1) / blockX;
    unsigned int gridY = ((unsigned int)height + blockY - 1) / blockY;
    void *args[] = {&gpu_ptr, &offsetX, &offsetY, &width, &height, &type};
    context.Run(cuLaunchKernel(main, gridX, gridY, 1, blockX, blockY, 1, 0, stream, args, NULL));
    if (enqueueDownload)
    {
        gpu_mem.CopyTo(pinned, stream, context);
        std::unique_ptr<streamCallbackData> data = make_unique<streamCallbackData>(
            &render_callback, pinned, (size_t)width, (size_t)height
        );
        context.Run(cuStreamAddCallback(stream, streamCallback, data.release(), 0));
    }
}
