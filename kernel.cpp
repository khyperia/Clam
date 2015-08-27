#include "kernel.h"
#include "kernelStructs.h"
#include "option.h"
#include "mandelbox.h"
#include "vrpn_help.h"
#include <fstream>

class KernelModuleBase
{
protected:
    virtual void Resize() = 0;

public:
    virtual ~KernelModuleBase()
    {
    }

    virtual bool Update(int w, int h) = 0;

    virtual bool OneTimeKeypress(SDL_Keycode keycode) = 0;

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time) = 0;

    virtual void SendState(StateSync *output, bool everything) const = 0;

    virtual bool RecvState(StateSync *input, bool everything) = 0;

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

    KernelModule(CUmodule module, const char *varname) : module(module), width(-1), height(-1)
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

    virtual bool Update(int w, int h)
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
                HandleCu(cuMemcpyHtoD(gpuVar, &value, sizeof(T)));
            }
            oldCpuVar = value;
        }
        return changed;
    }
};

template<typename T, typename Tscalar>
T CatmullRom(T p0, T p1, T p2, T p3, Tscalar t)
{
    Tscalar t2 = t * t;
    Tscalar t3 = t2 * t;

    return (((Tscalar)2 * p1) +
            (-p0 + p2) * t +
            ((Tscalar)2 * p0 - (Tscalar)5 * p1 + (Tscalar)4 * p2 - p3) * t2 +
            (-p0 + (Tscalar)3 * p1 - (Tscalar)3 * p2 + p3) * t3) / (Tscalar)2;
}

void Kernel::CommonOneTimeKeypress(SDL_Keycode keycode)
{
    if (keycode == SDLK_t)
    {
        std::cout << "Saving state" << std::endl;
        StateSync *sync = NewFileStateSync((Name() + ".clam3").c_str(), false);
        SendState(sync, false);
        delete sync;
    }
    else if (keycode == SDLK_y)
    {
        std::cout << "Loading state" << std::endl;
        StateSync *sync = NewFileStateSync((Name() + ".clam3").c_str(), true);
        RecvState(sync, false);
        delete sync;
    }
    else if (keycode == SDLK_x)
    {
        frame = frame == -1 ? 0 : -1;
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

class Module2dCamera : public KernelModule<Gpu2dCameraSettings>
{
    Vector2<double> pos;
    double zoom;
    VrpnHelp *vrpn;

public:
    Module2dCamera(CUmodule module) :
            KernelModule(module, "CameraArr"),
            pos(0, 0),
            zoom(1),
            vrpn(NULL)
    {
        ApplyParams();
    }

    ~Module2dCamera()
    {
        if (vrpn)
        {
            delete vrpn;
        }
    }

    virtual void Resize()
    {
        ApplyParams();
    }

    virtual bool OneTimeKeypress(SDL_Keycode keycode)
    {
        if (keycode == SDLK_BACKSLASH)
        {
            if (vrpn)
            {
                std::cout << "Disabling VRPN" << std::endl;
                delete vrpn;
                vrpn = NULL;
            }
            else
            {
                std::cout << "Enabling VRPN" << std::endl;
                vrpn = new VrpnHelp();
            }
        }
        return false;
    }

    virtual void Update()
    {
        if (IsUserInput() && vrpn)
        {
            vrpn->MainLoop();
            pos.x = vrpn->pos.x;
            pos.y = (vrpn->pos.z - 1.5) * -2;
            zoom = exp(vrpn->pos.y * -2);
        }
    }

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time)
    {
        const float movespeed = 0.5f;
        if (keycode == SDLK_w)
        {
            pos.y -= zoom * time * movespeed;
        }
        else if (keycode == SDLK_s)
        {
            pos.y += zoom * time * movespeed;
        }
        else if (keycode == SDLK_a)
        {
            pos.x -= zoom * time * movespeed;
        }
        else if (keycode == SDLK_d)
        {
            pos.x += zoom * time * movespeed;
        }
        else if (keycode == SDLK_i)
        {
            zoom /= 1 + time;
        }
        else if (keycode == SDLK_k)
        {
            zoom *= 1 + time;
        }
        else
        {
            return false;
        }
        ApplyParams();
        return true;
    }

    void ApplyParams()
    {
        value.posX = (float)pos.x;
        value.posY = (float)pos.y;
        value.zoom = (float)zoom;
    }

    virtual void SendState(StateSync *output, bool) const
    {
        output->Send(pos);
        output->Send(zoom);
    }

    virtual bool RecvState(StateSync *input, bool)
    {
        bool changed = false;
        changed |= input->RecvChanged(pos);
        changed |= input->RecvChanged(zoom);
        return changed;
    }

    virtual SDL_Surface *Configure(TTF_Font *) const
    {
        return NULL;
    }
};

class Module3dCamera : public KernelModule<GpuCameraSettings>
{
    Vector3<double> pos;
    Vector3<double> look;
    Vector3<double> up;
    double focalDistance;
    double fov;
    VrpnHelp *vrpn;

public:
    Module3dCamera(CUmodule module) :
            KernelModule(module, "CameraArr"),
            pos(10, 0, 0),
            look(-1, 0, 0),
            up(0, 1, 0),
            focalDistance(8),
            fov(1),
            vrpn(NULL)
    {
        ApplyParams();
    }

    ~Module3dCamera()
    {
        if (vrpn)
        {
            delete vrpn;
        }
    }

    virtual void Resize()
    {
        ApplyParams();
    }

    virtual bool OneTimeKeypress(SDL_Keycode keycode)
    {
        if (keycode == SDLK_BACKSLASH)
        {
            if (vrpn)
            {
                std::cout << "Disabling VRPN" << std::endl;
                delete vrpn;
                vrpn = NULL;
            }
            else
            {
                std::cout << "Enabling VRPN" << std::endl;
                vrpn = new VrpnHelp();
            }
        }
        return false;
    }

    virtual void Update()
    {
        if (IsUserInput() && vrpn)
        {
            vrpn->MainLoop();
            pos = vrpn->pos;
            look = vrpn->look;
            up = vrpn->up;
        }
    }

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time)
    {
        if (Common3dCamera(pos, look, up, focalDistance, fov, keycode, time))
        {
            ApplyParams();
            return true;
        }
        return false;
    }

    void ApplyParams()
    {
        look = look.normalized();
        up = cross(cross(look, up), look).normalized();
        value.posX = (float)pos.x;
        value.posY = (float)pos.y;
        value.posZ = (float)pos.z;
        value.lookX = (float)look.x;
        value.lookY = (float)look.y;
        value.lookZ = (float)look.z;
        value.upX = (float)up.x;
        value.upY = (float)up.y;
        value.upZ = (float)up.z;
        value.fov = (float)fov;
        value.focalDistance = (float)focalDistance;
    }

    virtual void SendState(StateSync *output, bool) const
    {
        output->Send(pos);
        output->Send(look);
        output->Send(up);
        output->Send(fov);
        output->Send(focalDistance);
    }

    virtual bool RecvState(StateSync *input, bool)
    {
        bool changed = false;
        changed |= input->RecvChanged(pos);
        changed |= input->RecvChanged(look);
        changed |= input->RecvChanged(up);
        changed |= input->RecvChanged(fov);
        changed |= input->RecvChanged(focalDistance);
        return changed;
    }

    virtual SDL_Surface *Configure(TTF_Font *) const
    {
        return NULL;
    }
};

class ModuleBuffer : public KernelModule<CUdeviceptr>
{
    int elementSize;

public:
    ModuleBuffer(CUmodule module, const char *varname, int elementSize) :
            KernelModule(module, varname),
            elementSize(elementSize)
    {
        value = 0;
    }

    ~ModuleBuffer()
    {
        if (value)
        {
            HandleCu(cuMemFree(value));
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

    virtual void SendState(StateSync *output, bool everything) const
    {
        if (everything)
        {
            size_t size = (size_t)elementSize * width * height;
            output->Send(size);
            output->SendArr(CuMem<char>(value, size).Download());
        }
    }

    virtual bool RecvState(StateSync *input, bool everything)
    {
        if (everything)
        {
            size_t size = input->Recv<size_t>();
            size_t existingSize = (size_t)elementSize * width * height;
            if (existingSize != size)
            {
                std::cout << "Not uploading state buffer due to mismatched sizes" << std::endl;
                input->RecvArr<char>(size);
                return false;
            }
            else
            {
                CuMem<char>(value, size).Upload(input->RecvArr<char>(size));
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

class ModuleMandelbox : public KernelModule<MandelboxCfg>
{
    int menuPos;

public:
    ModuleMandelbox(CUmodule module) :
            KernelModule(module, "MandelboxCfgArr")
    {
        value = MandelboxDefault();
    }

    virtual void Update()
    {
    }

    virtual void Resize()
    {
    }

    virtual bool OneTimeKeypress(SDL_Keycode keycode)
    {
        if (keycode == SDLK_UP)
        {
            menuPos--;
        }
        else if (keycode == SDLK_DOWN)
        {
            menuPos++;
        }
        else if (keycode == SDLK_LEFT)
        {
            MenuModI(-1);
        }
        else if (keycode == SDLK_RIGHT)
        {
            MenuModI(1);
        }
        const int max = 39;
        if (menuPos < 0)
        {
            menuPos = max - 1;
        }
        else if (menuPos >= max)
        {
            menuPos = 0;
        }
        return false;
    }

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time)
    {
        if (keycode == SDLK_LEFT)
        {
            MenuMod((float)-time);
        }
        else if (keycode == SDLK_RIGHT)
        {
            MenuMod((float)time);
        }
        else
        {
            return false;
        }
        return true;
    }

    virtual void SendState(StateSync *output, bool) const
    {
        output->Send(value);
    }

    virtual bool RecvState(StateSync *input, bool)
    {
        return input->RecvChanged(value);
    }

    void MenuModI(int delta)
    {
        switch (menuPos)
        {
            case 19:
                value.WhiteClamp = value.WhiteClamp == 0.0f ? 1.0f : 0.0f;
                break;
            case 30:
                value.MaxIters += delta;
                break;
            case 33:
                value.RandSeedInitSteps += delta;
                break;
            case 35:
                value.MaxRaySteps += delta;
                break;
            case 36:
                value.NumRayBounces += delta;
                break;
            default:
                return;
        }
    }

    void MenuMod(float delta)
    {
        switch (menuPos)
        {
            case 0:
                value.Scale += delta * 0.25f;
                break;
            case 1:
                value.FoldingLimit += delta * 0.25f;
                break;
            case 2:
                value.FixedRadius2 += delta * 0.25f;
                break;
            case 3:
                value.MinRadius2 += delta * 0.25f;
                break;
            case 4:
                value.InitRotation += delta * 0.25f;
                break;
            case 5:
                value.DeRotation += delta * 0.25f;
                break;
            case 6:
                value.ColorSharpness += delta * 1.0f;
                break;
            case 7:
                value.Saturation += delta * 0.2f;
                break;
            case 8:
                value.HueVariance += delta * 0.1f;
                break;
            case 9:
                value.Reflectivity += delta * 0.25f;
                break;
            case 10:
                value.DofAmount += delta * 0.01f;
                break;
            case 11:
                value.FovAbberation *= delta * 0.1f + 1.0f;
                break;
            case 12:
                value.LightPosX += delta * 0.5f;
                break;
            case 13:
                value.LightPosY += delta * 0.5f;
                break;
            case 14:
                value.LightPosZ += delta * 0.5f;
                break;
            case 15:
                value.LightSize *= delta * 0.1f + 1;
                break;
            case 16:
                value.ColorBiasR += delta * 0.25f;
                break;
            case 17:
                value.ColorBiasG += delta * 0.25f;
                break;
            case 18:
                value.ColorBiasB += delta * 0.25f;
                break;
            case 20:
                value.BrightThresh += delta * 0.1f;
                break;
            case 21:
                value.SpecularHighlightAmount += delta * 0.25f;
                break;
            case 22:
                value.SpecularHighlightSize += delta * 0.01f;
                break;
            case 23:
                value.FogDensity *= delta * 1.0f + 1;
                break;
            case 24:
                value.LightBrightnessAmount *= delta * 0.5f + 1;
                break;
            case 25:
                value.LightBrightnessCenter += delta * 0.25f;
                break;
            case 26:
                value.LightBrightnessWidth += delta * 0.25f;
                break;
            case 27:
                value.AmbientBrightnessAmount *= delta * 0.5f + 1;
                break;
            case 28:
                value.AmbientBrightnessCenter += delta * 0.25f;
                break;
            case 29:
                value.AmbientBrightnessWidth += delta * 0.25f;
                break;
            case 31:
                value.Bailout *= delta * 0.5f + 1;
                break;
            case 32:
                value.DeMultiplier += delta * 0.125f;
                break;
            case 34:
                value.MaxRayDist *= delta * 1.0f + 1;
                break;
            case 37:
                value.QualityFirstRay *= delta * 0.5f + 1;
                break;
            case 38:
                value.QualityRestRay *= delta * 0.5f + 1;
                break;
            default:
                return;
        }
        value.FogDensity = fabsf(value.FogDensity);
    }

    void MenuItem(TTF_Font *font, SDL_Surface *masterSurf, int &maxWidth, int &height, int index, float value,
                  const char *name) const
    {
        std::ostringstream result;
        result << (menuPos == index ? "* " : "  ") << name << " : " << value << "\n";
        if (masterSurf)
        {
            SDL_Color color = {255, 0, 0, 0};
            SDL_Surface *surf = TTF_RenderText_Blended(font, result.str().c_str(), color);
            SDL_Rect rect;
            rect.x = 0;
            rect.y = height;
            rect.w = maxWidth;
            rect.h = 5;
            SDL_BlitSurface(surf, NULL, masterSurf, &rect);
            SDL_FreeSurface(surf);
        }
        int thisWidth, thisHeight;
        TTF_SizeText(font, result.str().c_str(), &thisWidth, &thisHeight);
        if (thisWidth > maxWidth)
        {
            maxWidth = thisWidth;
        }
        height += thisHeight;
    }

    void Menu(TTF_Font *font, SDL_Surface *m, int &w, int &h) const
    {
        int i = 0;
        MenuItem(font, m, w, h, i++, value.Scale, "Scale");
        MenuItem(font, m, w, h, i++, value.FoldingLimit, "FoldingLimit");
        MenuItem(font, m, w, h, i++, value.FixedRadius2, "FixedRadius2");
        MenuItem(font, m, w, h, i++, value.MinRadius2, "MinRadius2");
        MenuItem(font, m, w, h, i++, value.InitRotation, "InitRotation");
        MenuItem(font, m, w, h, i++, value.DeRotation, "DeRotation");
        MenuItem(font, m, w, h, i++, value.ColorSharpness, "ColorSharpness");
        MenuItem(font, m, w, h, i++, value.Saturation, "Saturation");
        MenuItem(font, m, w, h, i++, value.HueVariance, "HueVariance");
        MenuItem(font, m, w, h, i++, value.Reflectivity, "Reflectivity");
        MenuItem(font, m, w, h, i++, value.DofAmount, "DofAmount");
        MenuItem(font, m, w, h, i++, value.FovAbberation, "FovAbberation");
        MenuItem(font, m, w, h, i++, value.LightPosX, "LightPosX");
        MenuItem(font, m, w, h, i++, value.LightPosY, "LightPosY");
        MenuItem(font, m, w, h, i++, value.LightPosZ, "LightPosZ");
        MenuItem(font, m, w, h, i++, value.LightSize, "LightSize");
        MenuItem(font, m, w, h, i++, value.ColorBiasR, "ColorBiasR");
        MenuItem(font, m, w, h, i++, value.ColorBiasG, "ColorBiasG");
        MenuItem(font, m, w, h, i++, value.ColorBiasB, "ColorBiasB");
        MenuItem(font, m, w, h, i++, value.WhiteClamp, "WhiteClamp");
        MenuItem(font, m, w, h, i++, value.BrightThresh, "BrightThresh");
        MenuItem(font, m, w, h, i++, value.SpecularHighlightAmount, "SpecularHighlightAmount");
        MenuItem(font, m, w, h, i++, value.SpecularHighlightSize, "SpecularHighlightSize");
        MenuItem(font, m, w, h, i++, value.FogDensity, "FogDensity");
        MenuItem(font, m, w, h, i++, value.LightBrightnessAmount, "LightBrightnessAmount");
        MenuItem(font, m, w, h, i++, value.LightBrightnessCenter, "LightBrightnessCenter");
        MenuItem(font, m, w, h, i++, value.LightBrightnessWidth, "LightBrightnessWidth");
        MenuItem(font, m, w, h, i++, value.AmbientBrightnessAmount, "AmbientBrightnessAmount");
        MenuItem(font, m, w, h, i++, value.AmbientBrightnessCenter, "AmbientBrightnessCenter");
        MenuItem(font, m, w, h, i++, value.AmbientBrightnessWidth, "AmbientBrightnessWidth");
        MenuItem(font, m, w, h, i++, value.MaxIters, "MaxIters");
        MenuItem(font, m, w, h, i++, value.Bailout, "Bailout");
        MenuItem(font, m, w, h, i++, value.DeMultiplier, "DeMultiplier");
        MenuItem(font, m, w, h, i++, value.RandSeedInitSteps, "RandSeedInitSteps");
        MenuItem(font, m, w, h, i++, value.MaxRayDist, "MaxRayDist");
        MenuItem(font, m, w, h, i++, value.MaxRaySteps, "MaxRaySteps");
        MenuItem(font, m, w, h, i++, value.NumRayBounces, "NumRayBounces");
        MenuItem(font, m, w, h, i++, value.QualityFirstRay, "QualityFirstRay");
        MenuItem(font, m, w, h, i++, value.QualityRestRay, "QualityRestRay");
    }

    virtual SDL_Surface *Configure(TTF_Font *font) const
    {
        int maxWidth = 0;
        int height = 0;
        Menu(font, NULL, maxWidth, height);
        SDL_Surface *master = SDL_CreateRGBSurface(SDL_SWSURFACE, maxWidth, height, 32, 255 << 16, 255 << 8, 255,
                                                   (Uint32)255 << 24);
        height = 0;
        Menu(font, master, maxWidth, height);
        return master;
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
        maxLocalSize(32)
{
    bool isCompute = IsCompute();
    if (name.empty())
    {
        name = "mandelbox";
        this->name = name;
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
        modules.push_back(new Module3dCamera(cuModule));
        modules.push_back(new ModuleMandelbox(cuModule));
        modules.push_back(new ModuleBuffer(cuModule, "BufferScratchArr", sizeof(float) * 4));
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
        modules.push_back(new Module2dCamera(cuModule));
    }
    else
    {
        throw std::runtime_error("Unknown kernel name " + name);
    }
    if (cuModule)
    {
        HandleCu(cuModuleGetFunction(&kernelMain, cuModule, "kern"));
    }
}

Kernel::~Kernel()
{
    if (cuModule)
    {
        HandleCu(cuModuleUnload(cuModule));
    }
    for (size_t i = 0; i < modules.size(); i++)
    {
        delete modules[i];
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
    for (size_t i = 0; i < modules.size(); i++)
    {
        modules[i]->SendState(output, everything);
    }
}

void Kernel::RecvState(StateSync *input, bool everything)
{
    for (size_t i = 0; i < modules.size(); i++)
    {
        if (modules[i]->RecvState(input, everything))
        {
            ResetFrame(frame);
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

void Kernel::SetTime(float time)
{
    std::cout << "SetTime not implemented: " << time << std::endl;
}

void Kernel::RenderInto(CuMem<int> &memory, size_t width, size_t height)
{
    //std::cout << "Resized from " << rngMemSize.x << "x" << rngMemSize.y <<
    //" to " << width << "x" << height << std::endl;
    for (size_t i = 0; i < modules.size(); i++)
    {
        if (modules[i]->Update((int)width, (int)height))
        {
            ResetFrame(frame);
        }
    }
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
                    &memory(),
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
    HandleCu(cuLaunchKernel(kernelMain, gridX, gridY, 1, blockX, blockY, 1, 0, NULL, args, NULL));
}

/*
class MandelboxAnimation
{
    std::vector<MandelboxState> keyframes;

public:
    void AddKeyframe(MandelboxState state)
    {
        keyframes.push_back(state);
    }

    void Clear()
    {
        keyframes.clear();
    }

    void Send(StateSync *sync)
    {
        sync->Send(keyframes.size());
        sync->SendArr(keyframes);
    }

    void Recv(StateSync *sync)
    {
        size_t size = sync->Recv<size_t>();
        keyframes = sync->RecvArr<MandelboxState>(size);
    }

    MandelboxState Interpolate(double time, bool loop)
    {
        time *= keyframes.size() - 1;
        size_t itime = (size_t)time;
        double t = time - itime;
        MandelboxState p0 = keyframes[itime == 0 ? (loop ? keyframes.size() - 1 : (size_t)0) : itime - 1];
        MandelboxState p1 = keyframes[itime];
        MandelboxState p2 = keyframes[itime + 1];
        MandelboxState p3 = keyframes[itime == keyframes.size() - 2 ? (loop ? 0 : itime + 1) : itime + 2];
        MandelboxState result;
#define InterpDo(ts, x) result.x = CatmullRom(p0.x, p1.x, p2.x, p3.x, (ts)t)
        InterpDo(double, pos);
        InterpDo(double, look);
        InterpDo(double, up);
        InterpDo(double, focalDistance);
        InterpDo(double, fov);
        InterpDo(float, cfg.Scale);
        InterpDo(float, cfg.FoldingLimit);
        InterpDo(float, cfg.FixedRadius2);
        InterpDo(float, cfg.MinRadius2);
        InterpDo(float, cfg.InitRotation);
        InterpDo(float, cfg.DeRotation);
        InterpDo(float, cfg.ColorSharpness);
        InterpDo(float, cfg.Saturation);
        InterpDo(float, cfg.HueVariance);
        InterpDo(float, cfg.Reflectivity);
        InterpDo(float, cfg.DofAmount);
        InterpDo(float, cfg.FovAbberation);
        InterpDo(float, cfg.LightPosX);
        InterpDo(float, cfg.LightPosY);
        InterpDo(float, cfg.LightPosZ);
        InterpDo(float, cfg.LightSize);
        InterpDo(float, cfg.ColorBiasR);
        InterpDo(float, cfg.ColorBiasG);
        InterpDo(float, cfg.ColorBiasB);
        InterpDo(float, cfg.WhiteClamp);
        InterpDo(float, cfg.BrightThresh);
        InterpDo(float, cfg.SpecularHighlightAmount);
        InterpDo(float, cfg.SpecularHighlightSize);
        InterpDo(float, cfg.FogDensity);
        InterpDo(float, cfg.LightBrightnessAmount);
        InterpDo(float, cfg.LightBrightnessCenter);
        InterpDo(float, cfg.LightBrightnessWidth);
        InterpDo(float, cfg.AmbientBrightnessAmount);
        InterpDo(float, cfg.AmbientBrightnessCenter);
        InterpDo(float, cfg.AmbientBrightnessWidth);
#undef InterpDo
        return result;
    }
};

    void SaveAnimation()
    {
        try
        {
            StateSync *sync = NewFileStateSync(AnimationName().c_str(), true);
            animation.Recv(sync);
            delete sync;
        }
        catch (const std::exception &ex)
        {
            std::cout << "Didn't read animation state: " << ex.what() << std::endl;
        }
        // ---
        StateSync *sync = NewFileStateSync(AnimationName().c_str(), false);
        animation.Send(sync);
        delete sync;
    }

    void OneTimeKeypress(SDL_Keycode keycode)
    {
        if (keycode == SDLK_v)
        {
            std::cout << "Saved keyframe" << std::endl;
            animation.AddKeyframe(state);
            SaveAnimation();
        }
        else if (keycode == SDLK_b)
        {
            std::cout << "Cleared keyframes" << std::endl;
            animation.Clear();
            SaveAnimation();
        }
        else
        {
            CommonOneTimeKeypress(this, keycode);
        }
    }
*/