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
        result.AddSetting("posX", 0.0);
        result.AddSetting("posY", 0.0);
        result.AddSetting("posZ", 5.0);
        result.AddSetting("lookX", 0.0);
        result.AddSetting("lookY", 0.0);
        result.AddSetting("lookZ", -1.0);
        result.AddSetting("upX", 0.0);
        result.AddSetting("upY", 1.0);
        result.AddSetting("upZ", 0.0);
        result.AddSetting("fov", 1.0);
        result.AddSetting("focalDistance", 3.0);
        result.AddSetting("Scale", -2.0);
        result.AddSetting("FoldingLimit", 1.0);
        result.AddSetting("FixedRadius2", 1.0);
        result.AddSetting("MinRadius2", 0.125);
        result.AddSetting("DeRotationAmount", 0.0);
        result.AddSetting("DeRotationAxisX", 1.0);
        result.AddSetting("DeRotationAxisY", 0.0);
        result.AddSetting("DeRotationAxisZ", 0.0);
        result.AddSetting("DofAmount", 0.01);
        result.AddSetting("FovAbberation", 0.0);
        result.AddSetting("LightPosX", 4.0);
        result.AddSetting("LightPosY", 4.0);
        result.AddSetting("LightPosZ", 4.0);
        result.AddSetting("LightSize", 1.0);
        result.AddSetting("WhiteClamp", false);
        result.AddSetting("LightBrightnessHue", 0.1);
        result.AddSetting("LightBrightnessSat", 0.7);
        result.AddSetting("LightBrightnessVal", 8.0);
        result.AddSetting("AmbientBrightnessHue", 0.6);
        result.AddSetting("AmbientBrightnessSat", 0.3);
        result.AddSetting("AmbientBrightnessVal", 1.0);
        result.AddSetting("ReflectHue", 0.0);
        result.AddSetting("ReflectSat", 0.0);
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
        result.AddSetting("posx", 0.0f);
        result.AddSetting("posy", 0.0f);
        result.AddSetting("zoom", 1.0f);
        result.AddSetting("juliax", 0.0f);
        result.AddSetting("juliay", 0.0f);
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
                                             1.0,
                                             SDL_SCANCODE_R,
                                             SDL_SCANCODE_F));
        result.push_back(make_unique<Camera3d>("posX",
                                               "posY",
                                               "posZ",
                                               "lookX",
                                               "lookY",
                                               "lookZ",
                                               "upX",
                                               "upY",
                                               "upZ",
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
        settings.push_back(std::make_pair("DeRotationAxisX", 0.25));
        settings.push_back(std::make_pair("DeRotationAxisY", 0.25));
        settings.push_back(std::make_pair("DeRotationAxisZ", 0.25));
        settings.push_back(std::make_pair("DofAmount", 0.01));
        settings.push_back(std::make_pair("FovAbberation", -0.1));
        settings.push_back(std::make_pair("LightPosX", 0.5));
        settings.push_back(std::make_pair("LightPosY", 0.5));
        settings.push_back(std::make_pair("LightPosZ", 0.5));
        settings.push_back(std::make_pair("LightSize", -0.2));
        settings.push_back(std::make_pair("WhiteClamp", 0));
        settings.push_back(std::make_pair("LightBrightnessHue", 0.125));
        settings.push_back(std::make_pair("LightBrightnessSat", -0.25));
        settings.push_back(std::make_pair("LightBrightnessVal", -0.25));
        settings.push_back(std::make_pair("AmbientBrightnessHue", 0.125));
        settings.push_back(std::make_pair("AmbientBrightnessSat", -0.25));
        settings.push_back(std::make_pair("AmbientBrightnessVal", -0.25));
        settings.push_back(std::make_pair("ReflectHue", 0.125));
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
        result.push_back(make_unique<Pan2d>("posx",
                                            "posy",
                                            "zoom",
                                            SDL_SCANCODE_W,
                                            SDL_SCANCODE_S,
                                            SDL_SCANCODE_A,
                                            SDL_SCANCODE_D));
        result.push_back(make_unique<Pan2d>("juliax",
                                            "juliay",
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
