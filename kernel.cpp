#include "kernel.h"
#include "option.h"
#include "vector.h"
#include "mandelbox.h"
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
        //if (pressedKeys.find(event.key.keysym.sym) == pressedKeys.end())
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

static CUmodule CommonBuild(const std::string &source)
{
    CUmodule result;
    HandleCu(cuModuleLoadData(&result, source.c_str()));
    return result;
}

extern const unsigned char mandelbox[];
extern const unsigned int mandelbox_len;

struct MandelboxState
{
    Vector3<double> pos;
    Vector3<double> look;
    Vector3<double> up;
    double focalDistance;
    double fov;
    MandelboxCfg cfg;

    MandelboxState() : pos(10, 0, 0), look(-1, 0, 0), up(0, 1, 0), focalDistance(8), fov(1), cfg(MandelboxDefault())
    {
    }

    bool operator==(const MandelboxState &right)
    {
        return memcmp(this, &right, sizeof(MandelboxState)) == 0;
    }

    bool operator!=(const MandelboxState &right)
    {
        return memcmp(this, &right, sizeof(MandelboxState)) != 0;
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

class MandelboxKernel : public Kernel
{
    CUmodule program;
    CUfunction kernelMain;
    CuMem<Vector2<int> > rngMem;
    CuMem<Vector4<float> > scratchMem;
    Vector2<size_t> rngMemSize;
    size_t maxLocalSize;
    MandelboxState state;
    int frame;
    MandelboxAnimation animation;
    Vector2<int> renderOffset;
    bool useRenderOffset;
    int menuPos;
    CUdeviceptr cuMandelboxCfg;
public:
    MandelboxKernel(bool isCompute)
            : rngMemSize(0, 0), state(),
              renderOffset(0, 0), useRenderOffset(RenderOffset(&renderOffset.x, &renderOffset.y))
    {
        if (isCompute)
        {
            program = CommonBuild(std::string((const char *)mandelbox, mandelbox_len));
            HandleCu(cuModuleGetFunction(&kernelMain, program, "kern"));
            size_t cfgSize;
            HandleCu(cuModuleGetGlobal(&cuMandelboxCfg, &cfgSize, program, "MandelboxCfgArr"));
            if (cfgSize != sizeof(MandelboxCfg))
            {
                throw std::runtime_error("MandelboxCfg CPU and GPU sizes don't agree");
            }
            maxLocalSize = 32;
            UpdateCfg();
        }
        else
        {
            program = 0;
            kernelMain = 0;
            maxLocalSize = 0;
        }
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
    }

    ~MandelboxKernel()
    {
        if (program)
        {
            HandleCu(cuModuleUnload(program));
        }
    };

    void UpdateCfg()
    {
        if (cuMandelboxCfg)
        {
            HandleCu(cuMemcpyHtoD(cuMandelboxCfg, &state.cfg, sizeof(MandelboxCfg)));
        }
    }

    void SaveAnimation()
    {
        StateSync *sync = NewFileStateSync(AnimationName().c_str(), false);
        animation.Send(sync);
        delete sync;
    }

    void OneTimeKeypress(SDL_Keycode keycode)
    {
        if (keycode == SDLK_UP)
        {
            menuPos--;
        }
        else if (keycode == SDLK_DOWN)
        {
            menuPos++;
        }
        else if (keycode == SDLK_v)
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

    void FixParams()
    {
        state.look = state.look.normalized();
        state.up = cross(cross(state.look, state.up), state.look).normalized();
    }

    void RepeatKeypress(SDL_Keycode keycode, double time)
    {
        if (keycode == SDLK_LEFT)
        {
            MenuMod((float)-time);
        }
        else if (keycode == SDLK_RIGHT)
        {
            MenuMod((float)time);
        }
        else if (Common3dCamera(state.pos, state.look, state.up, state.focalDistance, state.fov, keycode, time))
        {
            frame = 0;
            FixParams();
        }
    }

    void SendState(StateSync *output) const
    {
        output->Send(state);
    }

    void RecvState(StateSync *input)
    {
        bool changed = input->RecvChanged(state);
        if (changed)
        {
            frame = 0;
            FixParams();
            UpdateCfg();
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

    void RenderInto(CuMem<int> &memory, size_t width, size_t height)
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
        float posx = (float)state.pos.x;
        float posy = (float)state.pos.y;
        float posz = (float)state.pos.z;
        float lookx = (float)state.look.x;
        float looky = (float)state.look.y;
        float lookz = (float)state.look.z;
        float upx = (float)state.up.x;
        float upy = (float)state.up.y;
        float upz = (float)state.up.z;
        float myFov = (float)(state.fov * 2 / (width + height));
        float myFocalDistance = (float)state.focalDistance;
        float myFrame = (float)frame++;
        void *args[] =
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

    virtual void SetTime(float time)
    {
        state = animation.Interpolate(time, false);
        frame = 0;
        FixParams();
        UpdateCfg();
    }

    void MenuItem(TTF_Font *font, SDL_Surface *masterSurf, int &maxWidth, int &height, int index, float value,
                  const char *name)
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

    void MenuMod(float delta)
    {
        switch (menuPos)
        {
            case 0:
                state.cfg.Scale += delta * 0.25f;
                break;
            case 1:
                state.cfg.FoldingLimit += delta * 0.25f;
                break;
            case 2:
                state.cfg.FixedRadius2 += delta * 0.25f;
                break;
            case 3:
                state.cfg.MinRadius2 += delta * 0.25f;
                break;
            case 4:
                state.cfg.InitRotation += delta * 0.25f;
                break;
            case 5:
                state.cfg.DeRotation += delta * 0.25f;
                break;
            case 6:
                state.cfg.ColorSharpness += delta * 1.0f;
                break;
            case 7:
                state.cfg.Saturation += delta * 1.0f;
                break;
            case 8:
                state.cfg.HueVariance += delta * 0.1f;
                break;
            case 9:
                state.cfg.Reflectivity += delta * 0.25f;
                break;
            case 10:
                state.cfg.DofAmount += delta * 0.01f;
                break;
            case 11:
                state.cfg.FovAbberation += delta * 0.1f;
                break;
            case 12:
                state.cfg.LightPosX += delta * 0.5f;
                break;
            case 13:
                state.cfg.LightPosY += delta * 0.5f;
                break;
            case 14:
                state.cfg.LightPosZ += delta * 0.5f;
                break;
            case 15:
                state.cfg.LightSize += delta * 0.01f;
                break;
            case 16:
                state.cfg.ColorBiasR += delta * 0.25f;
                break;
            case 17:
                state.cfg.ColorBiasG += delta * 0.25f;
                break;
            case 18:
                state.cfg.ColorBiasB += delta * 0.25f;
                break;
            case 19:
                state.cfg.WhiteClamp += delta;
                break;
            case 20:
                state.cfg.BrightThresh += delta * 0.1f;
                break;
            case 21:
                state.cfg.SpecularHighlightAmount += delta * 0.25f;
                break;
            case 22:
                state.cfg.SpecularHighlightSize += delta * 0.01f;
                break;
            case 23:
                state.cfg.FogDensity += delta * 0.01f;
                break;
            case 24:
                state.cfg.LightBrightnessAmount += delta * 10.0f;
                break;
            case 25:
                state.cfg.LightBrightnessCenter += delta * 0.25f;
                break;
            case 26:
                state.cfg.LightBrightnessWidth += delta * 0.25f;
                break;
            case 27:
                state.cfg.AmbientBrightnessAmount += delta * 0.1f;
                break;
            case 28:
                state.cfg.AmbientBrightnessCenter += delta * 0.25f;
                break;
            case 29:
                state.cfg.AmbientBrightnessWidth += delta * 0.25f;
                break;
            default:
                return;
        }
        UpdateCfg();
        frame = 0;
    }

    void Menu(TTF_Font *font, SDL_Surface *m, int &w, int &h)
    {
        int i = 0;
        MenuItem(font, m, w, h, i++, state.cfg.Scale, "Scale");
        MenuItem(font, m, w, h, i++, state.cfg.FoldingLimit, "FoldingLimit");
        MenuItem(font, m, w, h, i++, state.cfg.FixedRadius2, "FixedRadius2");
        MenuItem(font, m, w, h, i++, state.cfg.MinRadius2, "MinRadius2");
        MenuItem(font, m, w, h, i++, state.cfg.InitRotation, "InitRotation");
        MenuItem(font, m, w, h, i++, state.cfg.DeRotation, "DeRotation");
        MenuItem(font, m, w, h, i++, state.cfg.ColorSharpness, "ColorSharpness");
        MenuItem(font, m, w, h, i++, state.cfg.Saturation, "Saturation");
        MenuItem(font, m, w, h, i++, state.cfg.HueVariance, "HueVariance");
        MenuItem(font, m, w, h, i++, state.cfg.Reflectivity, "Reflectivity");
        MenuItem(font, m, w, h, i++, state.cfg.DofAmount, "DofAmount");
        MenuItem(font, m, w, h, i++, state.cfg.FovAbberation, "FovAbberation");
        MenuItem(font, m, w, h, i++, state.cfg.LightPosX, "LightPosX");
        MenuItem(font, m, w, h, i++, state.cfg.LightPosY, "LightPosY");
        MenuItem(font, m, w, h, i++, state.cfg.LightPosZ, "LightPosZ");
        MenuItem(font, m, w, h, i++, state.cfg.LightSize, "LightSize");
        MenuItem(font, m, w, h, i++, state.cfg.ColorBiasR, "ColorBiasR");
        MenuItem(font, m, w, h, i++, state.cfg.ColorBiasG, "ColorBiasG");
        MenuItem(font, m, w, h, i++, state.cfg.ColorBiasB, "ColorBiasB");
        MenuItem(font, m, w, h, i++, state.cfg.WhiteClamp, "WhiteClamp");
        MenuItem(font, m, w, h, i++, state.cfg.BrightThresh, "BrightThresh");
        MenuItem(font, m, w, h, i++, state.cfg.SpecularHighlightAmount, "SpecularHighlightAmount");
        MenuItem(font, m, w, h, i++, state.cfg.SpecularHighlightSize, "SpecularHighlightSize");
        MenuItem(font, m, w, h, i++, state.cfg.FogDensity, "FogDensity");
        MenuItem(font, m, w, h, i++, state.cfg.LightBrightnessAmount, "LightBrightnessAmount");
        MenuItem(font, m, w, h, i++, state.cfg.LightBrightnessCenter, "LightBrightnessCenter");
        MenuItem(font, m, w, h, i++, state.cfg.LightBrightnessWidth, "LightBrightnessWidth");
        MenuItem(font, m, w, h, i++, state.cfg.AmbientBrightnessAmount, "AmbientBrightnessAmount");
        MenuItem(font, m, w, h, i++, state.cfg.AmbientBrightnessCenter, "AmbientBrightnessCenter");
        MenuItem(font, m, w, h, i++, state.cfg.AmbientBrightnessWidth, "AmbientBrightnessWidth");
        if (menuPos < 0)
        {
            menuPos = i - 1;
        }
        else if (menuPos >= i)
        {
            menuPos = 0;
        }
    }

    virtual SDL_Surface *Configure(TTF_Font *font)
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

    std::string Name()
    {
        return "mandelbox";
    }

    std::string AnimationName()
    {
        return Name() + ".animation.clam3";
    }
};

Kernel *MakeKernel()
{
    bool isCompute = IsCompute();
    std::string name = KernelName();
    if (name == "mandelbox")
    {
        return new MandelboxKernel(isCompute);
    }
    else if (name.empty())
    {
        std::cout << "Kernel not specified, defaulting to mandelbox" << std::endl;
        return new MandelboxKernel(isCompute);
    }
    else
    {
        throw std::runtime_error("Unknown kernel name " + name);
    }
}