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
        result.AddSetting("test", 1.0f);
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
