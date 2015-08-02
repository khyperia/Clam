#include "clcontext.h"
#include <SDL_video.h>
#include <stdexcept>
#include <sstream>

static cl::Platform GetPlatform(std::string platformName)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platformName.empty())
    {
        if (platforms.size() == 0)
        {
            throw std::runtime_error("No OpenCL platform found");
        }
        else if (platforms.size() == 1)
        {
            return platforms[0];
        }
        else
        {
            std::ostringstream str;
            str << "More than one default platforms exist. Available are:" << std::endl;
            for (size_t i = 0; i < platforms.size(); i++)
                str << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
            throw std::runtime_error(str.str());
        }
    }
    else
    {
        for (size_t i = 0; i < platforms.size(); i++)
        {
            if (platforms[i].getInfo<CL_PLATFORM_NAME>() == platformName)
            {
                return platforms[i];
            }
        }
        throw std::runtime_error("Could not find platform " + platformName);
    }
}

cl::Context GetDevice(std::string platformName, bool wireToDisplay)
{
    cl::Platform platform = GetPlatform(platformName);
    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
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
        int index = 0;
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
    return cl::Context(CL_DEVICE_TYPE_ALL, properties);
}
