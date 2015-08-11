#include "kernel.h"
#include "option.h"
#include "vector.h"
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

static CUmodule CommonBuild(const std::string &source)
{
    CUmodule result;
    HandleCu(cuModuleLoadData(&result, source.c_str()));
    return result;
}

extern const unsigned char mandelbox[];
extern const unsigned int mandelbox_len;

class MandelboxKernel : public Kernel
{
    CUmodule program;
    CUfunction kernelMain;
    CuMem<Vector2<int> > rngMem;
    CuMem<Vector4<float> > scratchMem;
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
    MandelboxKernel(bool isCompute)
            : rngMemSize(0, 0), pos(10, 0, 0), look(-1, 0, 0), up(0, 1, 0), focalDistance(8), fov(1),
              renderOffset(0, 0), useRenderOffset(RenderOffset(&renderOffset.x, &renderOffset.y))
    {
        if (isCompute)
        {
            program = CommonBuild(std::string((char *)mandelbox, mandelbox_len));
            HandleCu(cuModuleGetFunction(&kernelMain, program, "kern"));
            maxLocalSize = 32;
            TryDeserialize();
        }
        else
        {
            program = 0;
            kernelMain = 0;
            maxLocalSize = 0;
        }
    }

    ~MandelboxKernel()
    {
        if (kernelMain)
        {
            Serialize();
        }
        if (kernelMain)
        {
            // free kernelMain
        }
        if (program)
        {
            HandleCu(cuModuleUnload(program));
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
        output->SendArr(scratchMem.Download());
        output->SendArr(rngMem.Download());
    }

    void LoadWholeState(StateSync *input)
    {
        RecvState(input);
        frame = input->Recv<int>();
        rngMemSize.x = input->Recv<size_t>();
        rngMemSize.y = input->Recv<size_t>();
        size_t count = rngMemSize.x * rngMemSize.y;
        scratchMem = CuMem<Vector4<float> >::Upload(input->RecvArr<Vector4<float> >(count));
        rngMem = CuMem<Vector2<int> >::Upload(input->RecvArr<Vector2<int> >(count));
    }

    void RenderInto(CuMem<int>& memory, size_t width, size_t height)
    {
        if (width != rngMemSize.x || height != rngMemSize.y)
        {
            rngMem = CuMem<Vector2<int> >(width * height);
            scratchMem = CuMem<Vector4<float> >(width * height);
            frame = 0;
            std::cout << "Resized from " << rngMemSize.x << "x" << rngMemSize.y <<
            " to " << width << "x" << height << std::endl;
            rngMemSize = Vector2<size_t>(width, height);
        }
        int renderOffsetX = useRenderOffset ? renderOffset.x : -(int)width / 2;
        int renderOffsetY = useRenderOffset ? renderOffset.y : -(int)height / 2;
        int mywidth = (int)width;
        int myheight = (int)height;
        float posx = (float)pos.x;
        float posy = (float)pos.y;
        float posz = (float)pos.z;
        float lookx = (float)look.x;
        float looky = (float)look.y;
        float lookz = (float)look.z;
        float upx = (float)up.x;
        float upy = (float)up.y;
        float upz = (float)up.z;
        float myFov = (float)(fov * 2 / (width + height));
        float myFocalDistance = (float)focalDistance;
        float myFrame = (float)frame++;
        void* args[] =
                {
                        &memory(),
                        &scratchMem(),
                        &rngMem(),
                        &renderOffsetX,
                        &renderOffsetY,
                        &mywidth,
                        &myheight,
                        &posx,
                        &posy,
                        &posz,
                        &lookx,
                        &looky,
                        &lookz,
                        &upx,
                        &upy,
                        &upz,
                        &myFov,
                        &myFocalDistance,
                        &myFrame
                };
        unsigned int blockX = (unsigned int)maxLocalSize;
        unsigned int blockY = (unsigned int)maxLocalSize;
        unsigned int gridX = (unsigned int)(width + blockX - 1) / blockX;
        unsigned int gridY = (unsigned int)(height + blockY - 1) / blockY;
        HandleCu(cuLaunchKernel(kernelMain, gridX, gridY, 1, blockX, blockY, 1, 0, NULL, args, NULL));
    }

    std::string Name()
    {
        return "mandelbox";
    }
};

Kernel *MakeKernel()
{
    bool isCompute = IsCompute();
    static bool hasInit = false;
    if (isCompute && !hasInit)
    {
        hasInit = true;
        HandleCu(cuInit(0));
        CUdevice device;
        HandleCu(cuDeviceGet(&device, 0));
        CUcontext context;
        HandleCu(cuCtxCreate(&context, 0, device));
        HandleCu(cuCtxSetCurrent(context));
    }
    std::string name = KernelName();
    if (name == "mandelbox")
    {
        return new MandelboxKernel(isCompute);
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