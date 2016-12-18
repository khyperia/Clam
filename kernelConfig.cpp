#include "kernelConfig.h"

extern "C" {
extern const unsigned char mandelbox[];
extern const unsigned int mandelbox_len;

extern const unsigned char mandelbrot[];
extern const unsigned int mandelbrot_len;
}

KernelConfiguration::KernelConfiguration(const std::string &kernelName)
    : name(kernelName)
{
}

const unsigned char *KernelConfiguration::KernelData() const
{
    if (name == "mandelbox")
    {
        return mandelbox;
    }
    else if (name == "mandelbrot")
    {
        return mandelbrot;
    }
    throw std::runtime_error("Invalid kernel name");
}

unsigned int KernelConfiguration::KernelLength() const
{
    if (name == "mandelbox")
    {
        return mandelbox_len;
    }
    else if (name == "mandelbrot")
    {
        return mandelbrot_len;
    }
    throw std::runtime_error("Invalid kernel name");
}

SettingCollection KernelConfiguration::Settings() const
{
    SettingCollection result;
    if (name == "mandelbox")
    {
        result.AddSetting("pos", Vector3<double>(0.0, 0.0, 5.0));
        result.AddSetting("look", Vector3<double>(0.0, 0.0, -1.0));
        result.AddSetting("up", Vector3<double>(0.0, 1.0, 0.0));
        result.AddSetting("fov", 1.0);
        result.AddSetting("focalDistance", 3.0);
        result.AddSetting("Scale", -2.0);
        result.AddSetting("FoldingLimit", 1.0);
        result.AddSetting("FixedRadius2", 1.0);
        result.AddSetting("MinRadius2", 0.125);
        result.AddSetting("DeRotationAmount", 0.0);
        result.AddSetting("DeRotationAxis", Vector3<double>(1.0, 0.0, 0.0));
        result.AddSetting("DofAmount", 0.01);
        result.AddSetting("FovAbberation", 0.0);
        result.AddSetting("LightPos", Vector3<double>(4.0, 4.0, 4.0));
        result.AddSetting("LightSize", 1.0);
        result.AddSetting("WhiteClamp", false);
        result.AddSetting("LightBrightnessHue", 0.05);
        result.AddSetting("LightBrightnessSat", 0.7);
        result.AddSetting("LightBrightnessVal", 16.0);
        result.AddSetting("AmbientBrightnessHue", 0.6);
        result.AddSetting("AmbientBrightnessSat", 0.3);
        result.AddSetting("AmbientBrightnessVal", 1.0);
        result.AddSetting("ReflectHue", 0.0);
        result.AddSetting("ReflectSat", 0.001);
        result.AddSetting("ReflectVal", 1.0);
        result.AddSetting("MaxIters", 1024);
        result.AddSetting("Bailout", 1024.0);
        result.AddSetting("DeMultiplier", 0.95);
        result.AddSetting("RandSeedInitSteps", 64);
        result.AddSetting("MaxRayDist", 16.0);
        result.AddSetting("MaxRaySteps", 32);
        result.AddSetting("NumRayBounces", 3);
        result.AddSetting("QualityFirstRay", 2.0);
        result.AddSetting("QualityRestRay", 64.0);
        result.AddSetting("ItersPerKernel", 8);
    }
    else if (name == "mandelbrot")
    {
        result.AddSetting("pos", Vector2<double>(0.0, 0.0));
        result.AddSetting("zoom", 1.0f);
        result.AddSetting("julia", Vector2<double>(0.0, 0.0));
        result.AddSetting("juliaenabled", false);
    }
    else
    {
        throw std::runtime_error("Invalid kernel name");
    }
    return result;
}

std::vector<std::unique_ptr<KernelControl>>
KernelConfiguration::Controls(GpuKernel &kernel) const
{
    std::vector<std::unique_ptr<KernelControl>> result;
    if (name == "mandelbox")
    {
        result.push_back(make_unique<MandelboxKernelControl>(kernel.Variable(
            "CfgArr")));
    }
    else if (name == "mandelbrot")
    {
        result.push_back(make_unique<MandelbrotKernelControl>(kernel.Variable(
            "CfgArr")));
    }
    else
    {
        throw std::runtime_error("Invalid kernel name");
    }
    return result;
}

std::vector<std::unique_ptr<UiSetting>> KernelConfiguration::UiSettings() const
{
    std::vector<std::unique_ptr<UiSetting>> result;
    if (name == "mandelbox")
    {
        result.push_back(make_unique<ExpVar>("fov",
                                             1.0,
                                             SDL_SCANCODE_N,
                                             SDL_SCANCODE_M));
        result.push_back(make_unique<ExpVar>("focalDistance",
                                             0.25,
                                             SDL_SCANCODE_R,
                                             SDL_SCANCODE_F));
        result.push_back(make_unique<Camera3d>("pos",
                                               "look",
                                               "up",
                                               "fov",
                                               "focalDistance",
                                               SDL_SCANCODE_W,
                                               SDL_SCANCODE_S,
                                               SDL_SCANCODE_SPACE,
                                               SDL_SCANCODE_Z,
                                               SDL_SCANCODE_A,
                                               SDL_SCANCODE_D,
                                               SDL_SCANCODE_I,
                                               SDL_SCANCODE_K,
                                               SDL_SCANCODE_J,
                                               SDL_SCANCODE_L,
                                               SDL_SCANCODE_U,
                                               SDL_SCANCODE_O));
        std::vector<std::pair<std::string, double>> settings;
        settings.push_back(std::make_pair("Scale", 0.25));
        settings.push_back(std::make_pair("FoldingLimit", 0.25));
        settings.push_back(std::make_pair("FixedRadius2", 0.25));
        settings.push_back(std::make_pair("MinRadius2", 0.25));
        settings.push_back(std::make_pair("DeRotationAmount", 0.25));
        settings.push_back(std::make_pair("DeRotationAxis", 0.25));
        settings.push_back(std::make_pair("DofAmount", -1.0));
        settings.push_back(std::make_pair("FovAbberation", -0.1));
        settings.push_back(std::make_pair("LightPos", 0.5));
        settings.push_back(std::make_pair("LightSize", -0.2));
        settings.push_back(std::make_pair("WhiteClamp", 0));
        settings.push_back(std::make_pair("LightBrightnessHue", 0.05));
        settings.push_back(std::make_pair("LightBrightnessSat", -0.25));
        settings.push_back(std::make_pair("LightBrightnessVal", -0.25));
        settings.push_back(std::make_pair("AmbientBrightnessHue", 0.05));
        settings.push_back(std::make_pair("AmbientBrightnessSat", -0.25));
        settings.push_back(std::make_pair("AmbientBrightnessVal", -0.25));
        settings.push_back(std::make_pair("ReflectHue", 0.05));
        settings.push_back(std::make_pair("ReflectSat", -0.25));
        settings.push_back(std::make_pair("ReflectVal", -0.25));
        settings.push_back(std::make_pair("MaxIters", 0));
        settings.push_back(std::make_pair("Bailout", -0.5));
        settings.push_back(std::make_pair("DeMultiplier", 0.125));
        settings.push_back(std::make_pair("RandSeedInitSteps", 0));
        settings.push_back(std::make_pair("MaxRayDist", -1.0));
        settings.push_back(std::make_pair("MaxRaySteps", 0));
        settings.push_back(std::make_pair("NumRayBounces", 0));
        settings.push_back(std::make_pair("QualityFirstRay", -0.5));
        settings.push_back(std::make_pair("QualityRestRay", -0.5));
        settings.push_back(std::make_pair("ItersPerKernel", 0));
        result.push_back(make_unique<InteractiveSetting>(settings,
                                                         SDL_SCANCODE_UP,
                                                         SDL_SCANCODE_DOWN,
                                                         SDL_SCANCODE_RIGHT,
                                                         SDL_SCANCODE_LEFT));
    }
    else if (name == "mandelbrot")
    {
        result.push_back(make_unique<ExpVar>("zoom",
                                             1.0,
                                             SDL_SCANCODE_F,
                                             SDL_SCANCODE_R));
        result.push_back(make_unique<Pan2d>("pos",
                                            "zoom",
                                            SDL_SCANCODE_W,
                                            SDL_SCANCODE_S,
                                            SDL_SCANCODE_A,
                                            SDL_SCANCODE_D));
        result.push_back(make_unique<Pan2d>("julia",
                                            "zoom",
                                            SDL_SCANCODE_I,
                                            SDL_SCANCODE_K,
                                            SDL_SCANCODE_J,
                                            SDL_SCANCODE_L));
        result.push_back(make_unique<Toggle>("juliaenabled", SDL_SCANCODE_U));
    }
    else
    {
        throw std::runtime_error("Invalid kernel name");
    }
    return result;
}
