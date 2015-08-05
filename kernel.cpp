#include "kernel.h"
#include "option.h"
#include "clcontext.h"
#include "util.h"
#include "vector.h"
#include <iostream>
#include <fstream>

static void CommonOneTimeKeypress(Kernel *kern, SDL_Keycode keycode)
{
    if (keycode == SDLK_t)
    {
        std::cout << "Saving state" << std::endl;
        StateSync *sync = NewFileStateSync((kern->Name() + ".clam3").c_str(), false);
        kern->SendState(sync);
        delete sync;
    }
    else if (keycode == SDLK_y)
    {
        std::cout << "Loading state" << std::endl;
        StateSync *sync = NewFileStateSync((kern->Name() + ".clam3").c_str(), true);
        kern->RecvState(sync);
        delete sync;
    }
}

static bool Common3dCamera(
        Vector3<double> &pos,
        Vector3<double> &look,
        Vector3<double> &up,
        double &focalDistance,
        double &fov,
        SDL_Keycode keycode,
        double time)
{
    const double speed = 0.5;
    time *= speed;
    if (keycode == SDLK_w)
    {
        pos = pos + look * (time * focalDistance);
    }
    else if (keycode == SDLK_s)
    {
        pos = pos + look * (-time * focalDistance);
    }
    else if (keycode == SDLK_a)
    {
        pos = pos + cross(up, look) * (time * focalDistance);
    }
    else if (keycode == SDLK_d)
    {
        pos = pos + cross(look, up) * (time * focalDistance);
    }
    else if (keycode == SDLK_z)
    {
        pos = pos + up * (time * focalDistance);
    }
    else if (keycode == SDLK_SPACE)
    {
        pos = pos + up * (-time * focalDistance);
    }
    else if (keycode == SDLK_r)
    {
        focalDistance *= 1 + time * std::sqrt(fov);
    }
    else if (keycode == SDLK_f)
    {
        focalDistance /= 1 + time * std::sqrt(fov);
    }
    else if (keycode == SDLK_u || keycode == SDLK_q)
    {
        up = rotate(up, look, time);
    }
    else if (keycode == SDLK_o || keycode == SDLK_e)
    {
        up = rotate(up, look, -time);
    }
    else if (keycode == SDLK_j)
    {
        look = rotate(look, up, time * fov);
    }
    else if (keycode == SDLK_l)
    {
        look = rotate(look, up, -time * fov);
    }
    else if (keycode == SDLK_i)
    {
        look = rotate(look, cross(up, look), time * fov);
    }
    else if (keycode == SDLK_k)
    {
        look = rotate(look, cross(look, up), time * fov);
    }
    else if (keycode == SDLK_n)
    {
        fov *= 1 + time;
    }
    else if (keycode == SDLK_m)
    {
        fov /= 1 + time;
    }
    else
    {
        return false;
    }
    return true;
}

template<typename T>
std::vector<T> Download(cl_command_queue queue, cl_mem mem)
{
    size_t size;
    HandleCl(clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size), &size, NULL));
    std::vector<T> result(size / sizeof(T));
    clEnqueueReadBuffer(queue, mem, 1, 0, size, result.data(), 0, NULL, NULL);
    return result;
}

template<typename T>
cl_mem Upload(cl_context context, std::vector<T> data)
{
    cl_int err;
    cl_mem result = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   data.size() * sizeof(T), data.data(), &err);
    HandleCl(err);
    return result;
}

void Kernel::UserInput(SDL_Event event)
{
    if (event.type == SDL_KEYDOWN)
    {
        if (pressedKeys.find(event.key.keysym.sym) == pressedKeys.end())
        {
            OneTimeKeypress(event.key.keysym.sym);
        }
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
        RepeatKeypress(*iter, time);
    }
}

void Kernel::Serialize()
{
    std::string filename = Name() + ".autosave.clam3";
    StateSync *sync = NewFileStateSync(filename.c_str(), false);
    this->SaveWholeState(sync);
    delete sync;
    std::cout << "Saved progress to " << filename << std::endl;
}

void Kernel::TryDeserialize()
{
    std::string filename = Name() + ".autosave.clam3";
    try
    {
        StateSync *sync = NewFileStateSync(filename.c_str(), true);
        this->LoadWholeState(sync);
        delete sync;
        std::cout << "Loaded progress from " << filename << std::endl;
    }
    catch (const std::exception &err)
    {
        std::cout << "Failed to load progress from " << filename << std::endl;
    }
}

static std::vector<cl_device_id> GetContextDevices(cl_context context)
{
    size_t count;
    HandleCl(clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &count));
    std::vector<cl_device_id> result(count / sizeof(cl_device_id));
    HandleCl(clGetContextInfo(context, CL_CONTEXT_DEVICES, count, result.data(), NULL));
    return result;
}

static std::string DeviceName(cl_device_id device)
{
    size_t namesize;
    HandleCl(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &namesize));
    std::vector<char> result(namesize + 1);
    HandleCl(clGetDeviceInfo(device, CL_DEVICE_NAME, namesize, result.data(), NULL));
    for (size_t i = 0; i < result.size() - 1; i++)
    {
        if (result[i] == '\0')
        {
            result[i] = ' ';
        }
    }
    return std::string(result.data());
}

static std::string BuildLog(cl_program program, cl_device_id device)
{
    size_t logsize;
    HandleCl(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize));
    std::vector<char> result(logsize + 1);
    HandleCl(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize, result.data(), NULL));
    return std::string(result.data(), logsize);
}

static void DumpBin(cl_program program, std::string filename)
{
    size_t bincount;
    HandleCl(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 0, NULL, &bincount));
    std::vector<size_t> sizes(bincount / sizeof(size_t));
    HandleCl(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, bincount, sizes.data(), NULL));
    std::vector<char *> binaries(bincount / sizeof(size_t));
    for (size_t i = 0; i < bincount / sizeof(size_t); i++)
    {
        binaries[i] = new char[sizes[i]];
    }
    HandleCl(clGetProgramInfo(program, CL_PROGRAM_BINARIES, bincount / sizeof(size_t) * sizeof(char*), binaries.data(), NULL));
    for (size_t i = 0; i < bincount / sizeof(size_t); i++)
    {
        std::string outputName = i == 0 ? filename : filename + "." + tostring(i);
        std::ofstream output(outputName.c_str());
        output.write(binaries[i], sizes[i]);
        delete[] binaries[i];
        std::cout << "Dumped binary " << outputName << std::endl;
    }
}

static cl_program CommonBuild(cl_context context, const std::string &source)
{
    const char *sourceStr = source.data();
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &sourceStr, NULL, &err);
    HandleCl(err);
    err = clBuildProgram(program, 0, NULL,
                         "-cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -Werror",
                         NULL, NULL);

    std::vector<cl_device_id> devices = GetContextDevices(context);
    for (size_t i = 0; i < devices.size(); i++)
    {
        std::cout << "-- Build log for device " << DeviceName(devices[i]) << ":" << std::endl;
        std::cout << BuildLog(program, devices[i]) << std::endl;
    }
    std::cout << "-- End log. Compilation " << (err == CL_SUCCESS ? "succeeded" : "failed") << std::endl;
    std::string dumpbin = DumpBinary();
    if (!dumpbin.empty() && err == CL_SUCCESS)
    {
        DumpBin(program, dumpbin);
        throw std::runtime_error("Dumped binary, now exiting");
    }
    HandleCl(err);
    return program;
}

extern const unsigned char mandelbox[];
extern const unsigned int mandelbox_len;

class MandelboxKernel : public Kernel
{
    cl_program program;
    cl_kernel kernelMain;
    cl_mem rngMem;
    cl_mem scratchMem;
    Vector2<size_t> rngMemSize;
    size_t maxLocalSize;
    Vector3<double> pos;
    Vector3<double> look;
    Vector3<double> up;
    double focalDistance;
    double fov;
    int frame;
    Vector2<int> renderOffset;
    bool useRenderOffset;
public:
    MandelboxKernel(cl_context context)
            : rngMemSize(0, 0), pos(10, 0, 0), look(-1, 0, 0), up(0, 1, 0), focalDistance(8), fov(1),
              renderOffset(0, 0), useRenderOffset(RenderOffset(&renderOffset.x, &renderOffset.y))
    {
        this->context = context;
        if (context)
        {
            program = CommonBuild(context, std::string((char *)mandelbox, mandelbox_len));
            cl_int err;
            kernelMain = clCreateKernel(program, "kern", &err);
            HandleCl(err);

            std::vector<cl_device_id> devices = GetContextDevices(context);
            if (devices.size() != 1)
            {
                throw std::runtime_error("Mandelbox kernel only supports running on one device");
            }
            queue = clCreateCommandQueue(context, devices[0], 0, &err);
            HandleCl(err);

            HandleCl(clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                     sizeof(maxLocalSize), &maxLocalSize, NULL));
            TryDeserialize();
        }
    }

    ~MandelboxKernel()
    {
        if (context)
        {
            Serialize();
        }
        if (queue)
        {
            HandleCl(clReleaseCommandQueue(queue));
        }
        if (kernelMain)
        {
            HandleCl(clReleaseKernel(kernelMain));
        }
        if (program)
        {
            HandleCl(clReleaseProgram(program));
        }
        if (scratchMem)
        {
            HandleCl(clReleaseMemObject(scratchMem));
        }
        if (rngMem)
        {
            HandleCl(clReleaseMemObject(rngMem));
        }
    };

    void OneTimeKeypress(SDL_Keycode keycode)
    {
        CommonOneTimeKeypress(this, keycode);
    }

    void FixParams()
    {
        look = look.normalized();
        up = cross(cross(look, up), look).normalized();
    }

    void RepeatKeypress(SDL_Keycode keycode, double time)
    {
        if (Common3dCamera(pos, look, up, focalDistance, fov, keycode, time))
        {
            frame = 0;
            FixParams();
        }
    }

    void SendState(StateSync *output) const
    {
        output->Send(pos);
        output->Send(look);
        output->Send(up);
        output->Send(focalDistance);
        output->Send(fov);
    }

    void RecvState(StateSync *input)
    {
        bool changed = false;
        changed |= input->RecvChanged(pos);
        changed |= input->RecvChanged(look);
        changed |= input->RecvChanged(up);
        changed |= input->RecvChanged(focalDistance);
        changed |= input->RecvChanged(fov);
        if (changed)
        {
            frame = 0;
            FixParams();
        }
    }

    void SaveWholeState(StateSync *output) const
    {
        SendState(output);
        output->Send(frame);
        output->Send(rngMemSize.x);
        output->Send(rngMemSize.y);
        output->SendArr(Download<Vector4<float> >(queue, scratchMem));
        output->SendArr(Download<Vector4<float> >(queue, rngMem));
    }

    void LoadWholeState(StateSync *input)
    {
        RecvState(input);
        frame = input->Recv<int>();
        rngMemSize.x = input->Recv<size_t>();
        rngMemSize.y = input->Recv<size_t>();
        size_t count = rngMemSize.x * rngMemSize.y;
        scratchMem = Upload(context, input->RecvArr<Vector4<float> >(count));
        rngMem = Upload(context, input->RecvArr<Vector4<float> >(count));
    }

    template<typename T>
    void SetKernelArg(cl_kernel kernel, cl_uint index, const T &value)
    {
        clSetKernelArg(kernel, index, sizeof(value), &value);
    }

    void RenderInto(cl_mem memory, size_t width, size_t height)
    {
        if (width != rngMemSize.x || height != rngMemSize.y)
        {
            if (rngMem)
            {
                HandleCl(clReleaseMemObject(rngMem));
            }
            if (scratchMem)
            {
                HandleCl(clReleaseMemObject(scratchMem));
            }
            cl_int err;
            rngMem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 2 * width * height, NULL, &err);
            HandleCl(err);
            scratchMem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4 * width * height, NULL, &err);
            HandleCl(err);
            frame = 0;
            std::cout << "Resized from " << rngMemSize.x << "x" << rngMemSize.y <<
            " to " << width << "x" << height << std::endl;
            rngMemSize = Vector2<size_t>(width, height);
        }
        cl_uint index = 0;
        SetKernelArg(kernelMain, index++, memory);
        SetKernelArg(kernelMain, index++, scratchMem);
        SetKernelArg(kernelMain, index++, rngMem);
        SetKernelArg(kernelMain, index++, useRenderOffset ? renderOffset.x : -(int)width / 2);
        SetKernelArg(kernelMain, index++, useRenderOffset ? renderOffset.y : -(int)height / 2);
        SetKernelArg(kernelMain, index++, (int)width);
        SetKernelArg(kernelMain, index++, (int)height);
        SetKernelArg(kernelMain, index++, (float)pos.x);
        SetKernelArg(kernelMain, index++, (float)pos.y);
        SetKernelArg(kernelMain, index++, (float)pos.z);
        SetKernelArg(kernelMain, index++, (float)look.x);
        SetKernelArg(kernelMain, index++, (float)look.y);
        SetKernelArg(kernelMain, index++, (float)look.z);
        SetKernelArg(kernelMain, index++, (float)up.x);
        SetKernelArg(kernelMain, index++, (float)up.y);
        SetKernelArg(kernelMain, index++, (float)up.z);
        SetKernelArg(kernelMain, index++, (float)(fov * 2 / (width + height)));
        SetKernelArg(kernelMain, index++, (float)focalDistance);
        SetKernelArg(kernelMain, index++, (float)frame++);
        size_t localSize[] = {(size_t)std::sqrt(maxLocalSize), (size_t)std::sqrt(maxLocalSize)};
        size_t globalSize[] = {(width + localSize[0] - 1) / localSize[0] * localSize[0],
                               (height + localSize[1] - 1) / localSize[1] * localSize[1]};
        HandleCl(clEnqueueNDRangeKernel(queue, kernelMain, 2, NULL, globalSize, localSize, 0, NULL, NULL));
    }

    std::string Name()
    {
        return "mandelbox";
    }
};

Kernel *MakeKernel()
{
    cl_context context = 0;
    if (IsCompute())
    {
        context = GetDevice(PlatformName(), false);
    }
    std::string name = KernelName();
    if (name == "mandelbox")
    {
        return new MandelboxKernel(context);
    }
    else
    {
        if (name.empty())
        {
            throw std::runtime_error("Must specify a kernel");
        }
        throw std::runtime_error("Unknown kernel name " + name);
    }
}