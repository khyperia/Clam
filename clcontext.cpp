#include "clcontext.h"
#include "util.h"
#include <SDL_video.h>
#include <vector>
#include <CL/cl_gl.h>

static std::string PlatName(cl_platform_id platform)
{
    size_t namesize;
    HandleCl(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &namesize));
    std::vector<char> result(namesize + 1);
    HandleCl(clGetPlatformInfo(platform, CL_PLATFORM_NAME, namesize, result.data(), NULL));
    for (size_t i = 0; i < result.size() - 1; i++)
    {
        if (result[i] == '\0')
        {
            result[i] = ' ';
        }
    }
    return std::string(result.data());
}

static cl_platform_id GetPlatform(std::string platformName)
{
    cl_uint platCount;
    HandleCl(clGetPlatformIDs(0, NULL, &platCount));
    std::vector<cl_platform_id> platforms(platCount);
    HandleCl(clGetPlatformIDs((cl_uint)platforms.size(), platforms.data(), NULL));
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
            {
                str << PlatName(platforms[i]) << std::endl;
            }
            throw std::runtime_error(str.str());
        }
    }
    else
    {
        for (size_t i = 0; i < platforms.size(); i++)
        {
            if (PlatName(platforms[i]) == platformName)
            {
                return platforms[i];
            }
        }
        throw std::runtime_error("Could not find platform " + platformName);
    }
}

cl_context GetDevice(std::string platformName, bool wireToDisplay)
{
    cl_platform_id platform = GetPlatform(platformName);
    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            0, 0,
            0, 0,
            0, 0
    };
    if (wireToDisplay)
    {
        cl_context_properties (*getCurrentContext)(void) = (cl_context_properties (*)(void))SDL_GL_GetProcAddress(
                "glXGetCurrentContext");
        cl_context_properties (*getCurrentDisplay)(void) = (cl_context_properties (*)(void))SDL_GL_GetProcAddress(
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
    cl_int err;
    cl_context ctx = clCreateContextFromType(properties, CL_DEVICE_TYPE_ALL, NULL, NULL, &err);
    HandleCl(err);
    return ctx;
}
