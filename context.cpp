#include "context.h"
#include <stdexcept>
#include <GL/glx.h>

using namespace cl;

ClamContext::ClamContext()
{
    std::vector<Platform> platforms;
    Platform::get(&platforms);
    if (platforms.size() == 0)
    {
        throw std::runtime_error("No OpenCL platforms found");
    }

    cl_context_properties properties[] = {
#if defined (WIN32)
        CL_GL_CONTEXT_KHR , (cl_context_properties) wglGetCurrentContext() ,
        CL_WGL_HDC_KHR , (cl_context_properties) wglGetCurrentDC() ,
#elif defined (__linux__)
        CL_GL_CONTEXT_KHR , (cl_context_properties) glXGetCurrentContext() ,
        CL_GLX_DISPLAY_KHR , (cl_context_properties) glXGetCurrentDisplay() ,
#elif defined (__APPLE__)
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE , (cl_context_properties) CGLGetShareGroup( CGLGetCurrentContext() ) ,
#else
#error where the heck are we?
#endif
        CL_CONTEXT_PLATFORM , (cl_context_properties) platforms[0]() ,
        0 , 0 ,
    };
    context = std::make_shared<Context>(CL_DEVICE_TYPE_GPU, properties);

    std::vector<Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();
    if (devices.size() == 0)
    {
        throw std::runtime_error("No OpenCL devices found");
    }

    device = std::make_shared<cl::Device>(devices[0]);
}
