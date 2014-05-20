#include "context.h"
#include <stdexcept>
#include <GL/gl.h>
#include <CL/cl_gl.h>
#include <vector>

ClamContext::ClamContext()
{
	cl_uint openclPlatformCount = 0;
	clGetPlatformIDs(0, 0, &openclPlatformCount);
	if (openclPlatformCount == 0)
		throw std::runtime_error("No OpenCL platforms found");

	std::vector<cl_platform_id> openclPlatforms(openclPlatformCount);
	clGetPlatformIDs(openclPlatformCount, openclPlatforms.data(), 0);
	cl_platform_id openclPlatform = openclPlatforms[0];

	cl_uint openclDeviceCount = 0;
	clGetDeviceIDs(openclPlatform, CL_DEVICE_TYPE_GPU, 0, 0, &openclDeviceCount);
	if (openclDeviceCount == 0)
		throw std::runtime_error("No OpenCL devices found");

	std::vector<cl_device_id> openclDevices(openclDeviceCount);
	clGetDeviceIDs(openclPlatform, CL_DEVICE_TYPE_GPU, openclDeviceCount, openclDevices.data(), 0);

    cl_context_properties contextProperties[] = {
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
		CL_CONTEXT_PLATFORM, (cl_context_properties)openclPlatform,
        0 , 0 ,
    };

	cl_device_id openclDevice = 0;
	cl_context openclContext = 0;
	cl_int lError = 0;
	for (size_t i = 0; i < openclDevices.size(); ++i)
	{
		cl_device_id deviceIdToTry = openclDevices[i];
		cl_context contextToTry = 0;

		contextToTry = clCreateContext(
			contextProperties,
			1, &deviceIdToTry,
			0, 0,
			&lError
			);
		if (lError == CL_SUCCESS)
		{
			// We found the context.
			openclDevice = deviceIdToTry;
			openclContext = contextToTry;
			break;
		}
	}
	if (openclDevice == 0)
	{
		throw std::runtime_error("No compatible OpenCL devices found");
	}

	device = std::make_shared<cl_device_id>(openclDevice); // TODO: Dispose properly
	context = std::make_shared<cl_context>(openclContext); // TODO: Dispose properly
}