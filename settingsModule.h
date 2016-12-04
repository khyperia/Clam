#pragma once

#include "kernelStructs.h"
#include "vrpn_help.h"
#include "option.h"
#include "mandelbox.h"
#include "network.h"
#include <SDL.h>

class SettingModuleBase
{
public:
    virtual ~SettingModuleBase()
    {
    }

    virtual SettingModuleBase *Clone() const = 0;

    virtual const char *VarName() const = 0;

    virtual bool OneTimeKeypress(SDL_Keycode keycode) = 0;

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time) = 0;

    virtual void SendState(const StateSync *output) const = 0;

    virtual bool RecvState(const StateSync *input) = 0;

    virtual void Update() = 0;

    virtual void Animate(SettingModuleBase *zero,
                         SettingModuleBase *one,
                         SettingModuleBase *two,
                         SettingModuleBase *three,
                         double time) = 0;

    virtual SDL_Surface *Configure(TTF_Font *font) const = 0;
};

template<typename T>
struct SettingModule: public SettingModuleBase
{
    virtual ~SettingModule()
    {
    }

    virtual void Apply(T &value) const = 0;
};

class SettingAnimation
{
    std::vector<std::vector<SettingModuleBase *> > keyframes;

public:
    SettingAnimation(StateSync *sync,
                     const std::vector<SettingModuleBase *> &moduleSetup)
        : keyframes()
    {
        if (!sync)
        {
            return;
        }
        int length = sync->Recv<int>();
        std::cout << "Length " << length << std::endl;
        for (int i = 0; i < length; i++)
        {
            for (size_t mod = 0; mod < moduleSetup.size(); mod++)
            {
                moduleSetup[mod]->RecvState(sync);
            }
            AddKeyframe(moduleSetup);
            std::cout << "Added keyframe" << std::endl;
        }
    }

    ~SettingAnimation()
    {
        ClearKeyframes();
    }

    void ClearKeyframes()
    {
        for (size_t i = 0; i < keyframes.size(); i++)
        {
            for (size_t j = 0; j < keyframes[i].size(); j++)
            {
                delete keyframes[i][j];
            }
        }
        keyframes.clear();
    }

    void WriteKeyframes(StateSync *sync)
    {
        sync->Send((int)keyframes.size());
        for (size_t i = 0; i < keyframes.size(); i++)
        {
            for (size_t mod = 0; mod < keyframes[i].size(); mod++)
            {
                keyframes[i][mod]->SendState(sync);
            }
        }
    }

    void AddKeyframe(const std::vector<SettingModuleBase *> &currentState)
    {
        std::vector<SettingModuleBase *> cloned;
        for (size_t i = 0; i < currentState.size(); i++)
        {
            cloned.push_back(currentState[i]->Clone());
        }
        keyframes.push_back(cloned);
    }

    void Animate(std::vector<SettingModuleBase *> &currentState,
                 double time,
                 bool wrap)
    {
        if (keyframes.size() < 2)
        {
            throw new std::runtime_error("Not enough keyframes for animation");
        }
        time *= wrap ? keyframes.size() : keyframes.size() - 1;
        size_t index = (size_t)time;
        time -= index;
        size_t sizem1 = keyframes.size() - 1;
        const std::vector<SettingModuleBase *>
            &zero = keyframes[index == 0 ? (wrap ? sizem1 : 0) : index - 1];
        const std::vector<SettingModuleBase *> &one = keyframes[index];
        const std::vector<SettingModuleBase *>
            &two = keyframes[wrap ? (index + 1) % keyframes.size() : index + 1];
        const std::vector<SettingModuleBase *> &three =
            keyframes[wrap ? (index + 2) % keyframes.size() : (index + 1
                                                                   == sizem1
                                                               ? sizem1 : index
                                                                   + 2)];
        for (size_t i = 0; i < currentState.size(); i++)
        {
            currentState[i]->Animate(zero[i], one[i], two[i], three[i], time);
        }
    }
};

struct ModuleJuliaBrotSettings: public SettingModule<JuliaBrotSettings>
{
    Vector2<double> juliaPos;
    int juliaEnabled;
    double moveSpeed;

    ModuleJuliaBrotSettings()
        : juliaPos(0.5, 0.5), juliaEnabled(0), moveSpeed(0.5)
    {
    }

    virtual const char *VarName() const
    {
        return "JuliaArr";
    }

    virtual SettingModuleBase *Clone() const
    {
        ModuleJuliaBrotSettings *result = new ModuleJuliaBrotSettings();
        result->juliaPos = juliaPos;
        result->juliaEnabled = juliaEnabled;
        return result;
    }

    virtual void Update()
    {
    }

    virtual bool OneTimeKeypress(SDL_Keycode keycode)
    {
        if (keycode == SDLK_u || keycode == SDLK_o)
        {
            juliaEnabled = !juliaEnabled;
        }
        else
        {
            return false;
        }
        return true;
    }

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time)
    {
        if (juliaEnabled && keycode == SDLK_i)
        {
            juliaPos.y -= time * moveSpeed;
        }
        else if (juliaEnabled && keycode == SDLK_k)
        {
            juliaPos.y += time * moveSpeed;
        }
        else if (juliaEnabled && keycode == SDLK_j)
        {
            juliaPos.x -= time * moveSpeed;
        }
        else if (juliaEnabled && keycode == SDLK_l)
        {
            juliaPos.x += time * moveSpeed;
        }
        else if (juliaEnabled && keycode == SDLK_n)
        {
            moveSpeed *= 1 + time;
        }
        else if (juliaEnabled && keycode == SDLK_m)
        {
            moveSpeed /= 1 + time;
        }
        else
        {
            return false;
        }
        return true;
    }

    virtual void SendState(const StateSync *output) const
    {
        output->Send(juliaPos);
        output->Send(juliaEnabled);
    }

    virtual bool RecvState(const StateSync *input)
    {
        bool changed = false;
        changed |= input->RecvChanged(juliaPos);
        changed |= input->RecvChanged(juliaEnabled);
        return changed;
    }

    virtual void Apply(JuliaBrotSettings &value) const
    {
        value.juliaX = (float)juliaPos.x;
        value.juliaY = (float)juliaPos.y;
        value.juliaEnabled = juliaEnabled;
    }

    virtual void Animate(SettingModuleBase *zeroBase,
                         SettingModuleBase *oneBase,
                         SettingModuleBase *twoBase,
                         SettingModuleBase *threeBase,
                         double time)
    {
        ModuleJuliaBrotSettings
            *zero = dynamic_cast<ModuleJuliaBrotSettings *>(zeroBase);
        ModuleJuliaBrotSettings
            *one = dynamic_cast<ModuleJuliaBrotSettings *>(oneBase);
        ModuleJuliaBrotSettings
            *two = dynamic_cast<ModuleJuliaBrotSettings *>(twoBase);
        ModuleJuliaBrotSettings
            *three = dynamic_cast<ModuleJuliaBrotSettings *>(threeBase);
        juliaPos = CatmullRom(zero->juliaPos,
                              one->juliaPos,
                              two->juliaPos,
                              three->juliaPos,
                              time);
        juliaEnabled = one->juliaEnabled;
    }

    virtual SDL_Surface *Configure(TTF_Font *) const
    {
        return NULL;
    }
};

struct Module2dCameraSettings: public SettingModule<Gpu2dCameraSettings>
{
    Vector2<double> pos;
    double zoom;
    VrpnHelp *vrpn;

    Module2dCameraSettings()
        : pos(0, 0), zoom(1), vrpn(NULL)
    {
    }

    ~Module2dCameraSettings()
    {
        if (vrpn)
        {
            delete vrpn;
        }
    }

    virtual const char *VarName() const
    {
        return "CameraArr";
    }

    virtual SettingModuleBase *Clone() const
    {
        Module2dCameraSettings *result = new Module2dCameraSettings();
        result->pos = pos;
        result->zoom = zoom;
        return result;
    }

    virtual void Update()
    {
        if (IsUserInput() && vrpn)
        {
            vrpn->MainLoop();
            pos.x = vrpn->pos.x * 1;
            pos.y = (vrpn->pos.z - 1.5) * -1;
            zoom = exp(vrpn->pos.y);
        }
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
        else if (keycode == SDLK_r)
        {
            zoom /= 1 + time;
        }
        else if (keycode == SDLK_f)
        {
            zoom *= 1 + time;
        }
        else
        {
            return false;
        }
        return true;
    }

    virtual void SendState(const StateSync *output) const
    {
        output->Send(pos);
        output->Send(zoom);
    }

    virtual bool RecvState(const StateSync *input)
    {
        bool changed = false;
        changed |= input->RecvChanged(pos);
        changed |= input->RecvChanged(zoom);
        return changed;
    }

    virtual void Apply(Gpu2dCameraSettings &value) const
    {
        value.posX = (float)pos.x;
        value.posY = (float)pos.y;
        value.zoom = (float)zoom;
    }

    virtual void Animate(SettingModuleBase *zeroBase,
                         SettingModuleBase *oneBase,
                         SettingModuleBase *twoBase,
                         SettingModuleBase *threeBase,
                         double time)
    {
        Module2dCameraSettings
            *zero = dynamic_cast<Module2dCameraSettings *>(zeroBase);
        Module2dCameraSettings
            *one = dynamic_cast<Module2dCameraSettings *>(oneBase);
        Module2dCameraSettings
            *two = dynamic_cast<Module2dCameraSettings *>(twoBase);
        Module2dCameraSettings
            *three = dynamic_cast<Module2dCameraSettings *>(threeBase);
        pos = CatmullRom(zero->pos, one->pos, two->pos, three->pos, time);
        zoom = CatmullRom(zero->zoom, one->zoom, two->zoom, three->zoom, time);
    }

    virtual SDL_Surface *Configure(TTF_Font *) const
    {
        return NULL;
    }
};

struct Module3dCameraSettings: public SettingModule<GpuCameraSettings>
{
    Vector3<double> pos;
    Vector3<double> look;
    Vector3<double> up;
    double focalDistance;
    double fov;
    VrpnHelp *vrpn;

    Module3dCameraSettings()
        : pos(10, 0, 0), look(-1, 0, 0), up(0, 1, 0), focalDistance(8), fov(1),
          vrpn(NULL)
    {
    }

    ~Module3dCameraSettings()
    {
        if (vrpn)
        {
            delete vrpn;
        }
    }

    virtual const char *VarName() const
    {
        return "CameraArr";
    }

    virtual SettingModuleBase *Clone() const
    {
        Module3dCameraSettings *result = new Module3dCameraSettings();
        result->pos = pos;
        result->look = look;
        result->up = up;
        result->focalDistance = focalDistance;
        result->fov = fov;
        return result;
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

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time)
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
        look = look.normalized();
        up = cross(cross(look, up), look).normalized();
        return true;
    }

    virtual void SendState(const StateSync *output) const
    {
        output->Send(pos);
        output->Send(look);
        output->Send(up);
        output->Send(fov);
        output->Send(focalDistance);
    }

    virtual bool RecvState(const StateSync *input)
    {
        bool changed = false;
        changed |= input->RecvChanged(pos);
        changed |= input->RecvChanged(look);
        changed |= input->RecvChanged(up);
        changed |= input->RecvChanged(fov);
        changed |= input->RecvChanged(focalDistance);
        return changed;
    }

    virtual void Apply(GpuCameraSettings &value) const
    {
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

    virtual void Animate(SettingModuleBase *zeroBase,
                         SettingModuleBase *oneBase,
                         SettingModuleBase *twoBase,
                         SettingModuleBase *threeBase,
                         double time)
    {
        Module3dCameraSettings
            *zero = dynamic_cast<Module3dCameraSettings *>(zeroBase);
        Module3dCameraSettings
            *one = dynamic_cast<Module3dCameraSettings *>(oneBase);
        Module3dCameraSettings
            *two = dynamic_cast<Module3dCameraSettings *>(twoBase);
        Module3dCameraSettings
            *three = dynamic_cast<Module3dCameraSettings *>(threeBase);
        pos = CatmullRom(zero->pos, one->pos, two->pos, three->pos, time);
        look = CatmullRom(zero->look, one->look, two->look, three->look, time);
        up = CatmullRom(zero->up, one->up, two->up, three->up, time);
        fov = CatmullRom(zero->fov, one->fov, two->fov, three->fov, time);
        focalDistance = CatmullRom(zero->focalDistance,
                                   one->focalDistance,
                                   two->focalDistance,
                                   three->focalDistance,
                                   time);
        look = look.normalized();
        up = cross(cross(look, up), look).normalized();
    }

    virtual SDL_Surface *Configure(TTF_Font *) const
    {
        return NULL;
    }
};

struct ModuleMandelboxSettings: public SettingModule<MandelboxCfg>
{
    struct NameValue
    {
        virtual ~NameValue()
        {
        }

        virtual void Modify(double time, bool oneTime) = 0;

        virtual void MenuItem(TTF_Font *font,
                              SDL_Surface *masterSurf,
                              int &maxWidth,
                              int &height,
                              int myIndex,
                              int menuPos) const = 0;

        virtual void Animate(NameValue *one,
                             NameValue *two,
                             NameValue *three,
                             NameValue *four,
                             double time) = 0;
    };

    template<typename T>
    struct TypedNameValue: public NameValue
    {
        const char *name;
        T &value;
        T incType;

        TypedNameValue(const char *name, T &value, T incType)
            : name(name), value(value), incType(incType)
        {
        }

        virtual void Modify(double time, bool oneTime)
        {
            if (incType < 0)
            {
                if (!oneTime)
                {
                    value *= (T)std::exp((T)time * -incType);
                }
            }
            else if (incType > 0)
            {
                if (!oneTime)
                {
                    value += (T)time * incType;
                }
            }
            else
            {
                if (oneTime)
                {
                    value += (time > 0 ? 1 : -1);
                }
            }
        }

        virtual void MenuItem(TTF_Font *font,
                              SDL_Surface *masterSurf,
                              int &maxWidth,
                              int &height,
                              int myIndex,
                              int menuPos) const
        {
            std::ostringstream result;
            result << (menuPos == myIndex ? "* " : "  ") << name << " : "
                   << value << "\n";
            if (masterSurf)
            {
                SDL_Color color = {255, 0, 0, 0};
                SDL_Surface *surf =
                    TTF_RenderText_Blended(font, result.str().c_str(), color);
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

        virtual void Animate(NameValue *zeroBase,
                             NameValue *oneBase,
                             NameValue *twoBase,
                             NameValue *threeBase,
                             double time)
        {
            TypedNameValue<T>
                *zero = dynamic_cast<TypedNameValue<T> *>(zeroBase);
            TypedNameValue<T> *one = dynamic_cast<TypedNameValue<T> *>(oneBase);
            TypedNameValue<T> *two = dynamic_cast<TypedNameValue<T> *>(twoBase);
            TypedNameValue<T>
                *three = dynamic_cast<TypedNameValue<T> *>(threeBase);
            value = CatmullRom(zero->value,
                               one->value,
                               two->value,
                               three->value,
                               (T)time);
        }
    };

    int menuPos;
    MandelboxCfg editValue;
    std::vector<NameValue *> nameValues;

    ModuleMandelboxSettings()
    {
        menuPos = 0;
        editValue = MandelboxDefault();
#define MkNv(t, x, d) nameValues.push_back(new TypedNameValue<t>(#x, editValue.x, d))
        MkNv(float, Scale, 0.25f);
        MkNv(float, FoldingLimit, 0.25f);
        MkNv(float, FixedRadius2, 0.25f);
        MkNv(float, MinRadius2, 0.25f);
        MkNv(float, DeRotationAmount, 0.25f);
        MkNv(float, DeRotationAxisX, 0.25f);
        MkNv(float, DeRotationAxisY, 0.25f);
        MkNv(float, DeRotationAxisZ, 0.25f);
        MkNv(float, DofAmount, 0.01f);
        MkNv(float, FovAbberation, -0.1f);
        MkNv(float, LightPosX, 0.5f);
        MkNv(float, LightPosY, 0.5f);
        MkNv(float, LightPosZ, 0.5f);
        MkNv(float, LightSize, -0.2f);
        MkNv(int, WhiteClamp, 0);
        MkNv(float, LightBrightnessHue, 0.125f);
        MkNv(float, LightBrightnessSat, -0.25f);
        MkNv(float, LightBrightnessVal, -0.25f);
        MkNv(float, AmbientBrightnessHue, 0.125f);
        MkNv(float, AmbientBrightnessSat, -0.25f);
        MkNv(float, AmbientBrightnessVal, -0.25f);
        MkNv(float, ReflectHue, 0.125f);
        MkNv(float, ReflectSat, -0.25f);
        MkNv(float, ReflectVal, -0.25f);
        MkNv(int, MaxIters, 0);
        MkNv(float, Bailout, -0.5f);
        MkNv(float, DeMultiplier, 0.125f);
        MkNv(int, RandSeedInitSteps, 0);
        MkNv(float, MaxRayDist, -1.0f);
        MkNv(int, MaxRaySteps, 0);
        MkNv(int, NumRayBounces, 0);
        MkNv(float, QualityFirstRay, -0.5f);
        MkNv(float, QualityRestRay, -0.5f);
        MkNv(int, ItersPerKernel, 0);
#undef MkNv
    }

    ~ModuleMandelboxSettings()
    {
        for (size_t i = 0; i < nameValues.size(); i++)
        {
            delete nameValues[i];
        }
    }

    virtual const char *VarName() const
    {
        return "MandelboxCfgArr";
    }

    virtual SettingModuleBase *Clone() const
    {
        ModuleMandelboxSettings *result = new ModuleMandelboxSettings();
        result->editValue = editValue;
        return result;
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
            nameValues[menuPos]->Modify(-1, true);
            return true;
        }
        else if (keycode == SDLK_RIGHT)
        {
            nameValues[menuPos]->Modify(1, true);
            return true;
        }
        else if (keycode == SDLK_g)
        {
            editValue.DofAmount = editValue.DofAmount > 0 ? 0 : 0.005f;
            return true;
        }
        else
        {
            return false;
        }
        int sizeMinusOne = (int)(nameValues.size() - 1);
        if (menuPos < 0)
        {
            menuPos = sizeMinusOne;
        }
        if (menuPos > sizeMinusOne)
        {
            menuPos = 0;
        }
        return false;
    }

    virtual bool RepeatKeypress(SDL_Keycode keycode, double time)
    {
        if (keycode == SDLK_LEFT)
        {
            nameValues[menuPos]->Modify(-time, false);
        }
        else if (keycode == SDLK_RIGHT)
        {
            nameValues[menuPos]->Modify(time, false);
        }
        else
        {
            return false;
        }
        return true;
    }

    virtual void SendState(const StateSync *output) const
    {
        output->Send(editValue);
    }

    virtual bool RecvState(const StateSync *input)
    {
        return input->RecvChanged(editValue);
    }

    virtual void Update()
    {
    }

    virtual void Apply(MandelboxCfg &value) const
    {
        value = editValue;
    }

    virtual void Animate(SettingModuleBase *zeroBase,
                         SettingModuleBase *oneBase,
                         SettingModuleBase *twoBase,
                         SettingModuleBase *threeBase,
                         double time)
    {
        ModuleMandelboxSettings
            *zero = dynamic_cast<ModuleMandelboxSettings *>(zeroBase);
        ModuleMandelboxSettings
            *one = dynamic_cast<ModuleMandelboxSettings *>(oneBase);
        ModuleMandelboxSettings
            *two = dynamic_cast<ModuleMandelboxSettings *>(twoBase);
        ModuleMandelboxSettings
            *three = dynamic_cast<ModuleMandelboxSettings *>(threeBase);
        for (size_t i = 0; i < nameValues.size(); i++)
        {
            nameValues[i]->Animate(zero->nameValues[i],
                                   one->nameValues[i],
                                   two->nameValues[i],
                                   three->nameValues[i],
                                   time);
        }
    }

    virtual SDL_Surface *Configure(TTF_Font *font) const
    {
        int maxWidth = 0;
        int height = 0;
        for (size_t i = 0; i < nameValues.size(); i++)
        {
            nameValues[i]
                ->MenuItem(font, NULL, maxWidth, height, (int)i, menuPos);
        }
        SDL_Surface *master = SDL_CreateRGBSurface(SDL_SWSURFACE,
                                                   maxWidth,
                                                   height,
                                                   32,
                                                   255 << 16,
                                                   255 << 8,
                                                   255,
                                                   (Uint32)255 << 24);
        height = 0;
        for (size_t i = 0; i < nameValues.size(); i++)
        {
            nameValues[i]
                ->MenuItem(font, master, maxWidth, height, (int)i, menuPos);
        }
        return master;
    }
};
