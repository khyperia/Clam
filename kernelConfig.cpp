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
    }
    else if (name == "mandelbrot")
    {
        result.push_back(make_unique<MandelbrotKernelControl>(kernel.Variable(
            "CameraArr")));
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
        result.push_back(make_unique<Camera2d>());
    }
    else
    {
        throw std::runtime_error("Invalid kernel name");
    }
    return result;
}
