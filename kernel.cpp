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
std::vector<T> Download(const cl::CommandQueue &queue, const cl::Buffer &mem)
{
    size_t size = mem.getInfo<CL_MEM_SIZE>();
    std::vector<T> result(size / sizeof(T));
    queue.enqueueReadBuffer(mem, 1, 0, size, result.data());
    return result;
}

template<typename T>
cl::Buffer Upload(const cl::Context &context, std::vector<T> data)
{
    return cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      data.size() * sizeof(T), data.data());
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

static cl::Program CommonBuild(const cl::Context &context, const std::string &source)
{
    cl::Program program = cl::Program(context, source);
    bool success = false;
    try
    {
        cl_int result = program.build(
                "-cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -cl-uniform-work-group-size -Werror",
                NULL, NULL);
        success = result == 0;
    }
    catch (const std::exception &ex)
    {
        success = false;
    }

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    for (size_t i = 0; i < devices.size(); i++)
    {
        std::cout << "-- Build log for device " << devices[i].getInfo<CL_DEVICE_NAME>() << ":" << std::endl;
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i]) << std::endl;
    }
    std::cout << "-- End log. Compilation " << (success ? "succeeded" : "failed") << std::endl;
    std::string dumpbin = DumpBinary();
    if (!dumpbin.empty() && success)
    {
        std::vector<size_t> sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
        std::vector<char *> binary = program.getInfo<CL_PROGRAM_BINARIES>();
        for (size_t i = 0; i < binary.size(); i++)
        {
            std::string outputName = i == 0 ? dumpbin : dumpbin + "." + tostring(i);
            std::ofstream output(outputName.c_str());
            output.write(binary[i], sizes[i]);
            std::cout << "Dumped binary " << outputName << std::endl;
        }
    }
    return program;
}

extern unsigned char mandelbox_cl[];
extern unsigned int mandelbox_cl_len;

class MandelboxKernel : public Kernel
{
    cl::Program program;
    cl::Kernel kernelMain;
    cl::Buffer rngMem;
    cl::Buffer scratchMem;
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
    MandelboxKernel(cl::Context context)
            : rngMemSize(0, 0), pos(10, 0, 0), look(-1, 0, 0), up(0, 1, 0), focalDistance(8), fov(1),
              renderOffset(0, 0), useRenderOffset(RenderOffset(&renderOffset.x, &renderOffset.y))
    {
        this->context = context;
        if (context())
        {
            program = CommonBuild(context, std::string((char *)mandelbox_cl, mandelbox_cl_len));
            kernelMain = cl::Kernel(program, "kern");
            std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
            if (devices.size() != 1)
            {
                throw std::runtime_error("Mandelbox kernel only supports running on one device");
            }
            queue = cl::CommandQueue(context, devices[0]);
            maxLocalSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            TryDeserialize();
        }
    }

    ~MandelboxKernel()
    {
        if (context())
        {
            Serialize();
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

    void RenderInto(cl::Memory memory, size_t width, size_t height)
    {
        if (width != rngMemSize.x || height != rngMemSize.y)
        {
            rngMem = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * 2 * width * height);
            scratchMem = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4 * width * height);
            std::cout << "Resized from " << rngMemSize.x << "x" << rngMemSize.y <<
            " to " << width << "x" << height << std::endl;
            rngMemSize = Vector2<size_t>(width, height);
        }
        cl_uint index = 0;
        kernelMain.setArg(index++, memory);
        kernelMain.setArg(index++, scratchMem);
        kernelMain.setArg(index++, rngMem);
        kernelMain.setArg(index++, useRenderOffset ? renderOffset.x : -(int)width / 2);
        kernelMain.setArg(index++, useRenderOffset ? renderOffset.y : -(int)height / 2);
        kernelMain.setArg(index++, (int)width);
        kernelMain.setArg(index++, (int)height);
        kernelMain.setArg(index++, (float)pos.x);
        kernelMain.setArg(index++, (float)pos.y);
        kernelMain.setArg(index++, (float)pos.z);
        kernelMain.setArg(index++, (float)look.x);
        kernelMain.setArg(index++, (float)look.y);
        kernelMain.setArg(index++, (float)look.z);
        kernelMain.setArg(index++, (float)up.x);
        kernelMain.setArg(index++, (float)up.y);
        kernelMain.setArg(index++, (float)up.z);
        kernelMain.setArg(index++, (float)(fov * 2 / (width + height)));
        kernelMain.setArg(index++, (float)focalDistance);
        kernelMain.setArg(index++, (float)frame++);
        size_t localWidth = maxLocalSize;
        size_t localHeight = 1;
        cl::NDRange localSize(localWidth, localHeight);
        cl::NDRange globalSize((width + localWidth - 1) / localWidth * localWidth,
                               (height + localHeight - 1) / localHeight * localHeight);
        queue.enqueueNDRangeKernel(kernelMain, cl::NDRange(0, 0), globalSize, localSize);
    }

    std::string Name()
    {
        return "mandelbox";
    }
};

Kernel *MakeKernel()
{
    cl::Context context;
    if (IsCompute())
    {
        context = GetDevice(PlatformName(), DeviceName(), false);
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