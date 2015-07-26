#include "clcontext.h"
#include "util.h"
#include <SDL_video.h>

static cl::Context CreateDevice(const cl::Platform &platform, const cl::Device &device, bool wireToDisplay)
{
    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties) platform(),
            0, 0,
            0, 0,
            0, 0
    };
    if (wireToDisplay)
    {
        cl_context_properties (*getCurrentContext)(void) = (cl_context_properties (*)(void)) SDL_GL_GetProcAddress(
                "glXGetCurrentContext");
        cl_context_properties (*getCurrentDisplay)(void) = (cl_context_properties (*)(void)) SDL_GL_GetProcAddress(
                "glXGetCurrentDisplay");
        int index = 2;
        if (getCurrentContext)
        {
            cl_context_properties currentContext = getCurrentContext();
            if (currentContext)
            {
                properties[index++] = CL_GL_CONTEXT_KHR;
                properties[index++] = currentContext;
            }
        }
        if (getCurrentDisplay)
        {
            cl_context_properties currentDisplay = getCurrentDisplay();
            if (currentDisplay)
            {
                properties[index++] = CL_GLX_DISPLAY_KHR;
                properties[index++] = currentDisplay;
            }
        }
    }
    return cl::Context(device, properties);
}

cl::Context GetDevice(std::string platname, std::string devname, bool wireToDisplay)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::ostringstream errmessage;
    errmessage << "Could not find device \"" << platname << "\", \"" << devname << "\". Available devices are:\n";
    for (size_t platformIdx = 0; platformIdx < platforms.size(); platformIdx++)
    {
        cl::Platform &platform = platforms[platformIdx];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        std::vector<cl::Device> devices;
        if (platform.getDevices(CL_DEVICE_TYPE_ALL, &devices))
        {
            errmessage << "(note: Could not get devices for \"" << platformName << "\")\n";
            continue;
        }
        for (size_t deviceIdx = 0; deviceIdx < devices.size(); deviceIdx++)
        {
            cl::Device &device = devices[deviceIdx];
            std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
            if ((platname.empty() || platname == platformName) &&
                devname == deviceName)
            {
                return CreateDevice(platform, device, wireToDisplay);
            }
            errmessage << "\"" << platformName << "\", \"" << deviceName << "\"\n";
        }
    }
    throw std::runtime_error(errmessage.str());
}
