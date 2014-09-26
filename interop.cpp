#include "interop.h"
#include "helper.h"
#include <CL/cl_gl.h>

ClamInterop::ClamInterop()
{
}

ClamInterop::ClamInterop(std::shared_ptr<cl_context> context)
	: clContext(context)
{
	glClearColor(1.f, 0.f, 0.f, 1.f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.f, 1.f, 0.f, 1.f, -1.f, 1.f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	destructor = std::make_shared<Destructor>();
	glGenTextures(1, &destructor->glTexture);
	glGenBuffers(1, &destructor->glBuffer);

	glEnable(GL_TEXTURE_2D);
}

ClamInterop::Destructor::~Destructor()
{
	glDeleteTextures(1, &glTexture);
	glDeleteBuffers(1, &glBuffer);
}

void DeleteReleaseMemObject(cl_mem* ptr)
{
    clReleaseMemObject(*ptr);
    delete ptr;
}

void ClamInterop::Resize(std::shared_ptr<ClamKernel> kernel, int _width, int _height)
{
    if (_width == 0 || _height == 0)
        return;
	width = _width;
	height = _height;
	glViewport(0, 0, width, height);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, destructor->glBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(float), nullptr, GL_DYNAMIC_COPY);
    
    if (clBuffers.find("") != clBuffers.end())
        clReleaseMemObject(*clBuffers[""]);
	cl_int openclError = 0;
	cl_mem openclBuffer = clCreateFromGLBuffer(*clContext,
            CL_MEM_WRITE_ONLY, destructor->glBuffer, &openclError);
	clBuffers[""] = std::make_shared<cl_mem>(openclBuffer);
	
	if (openclError)
		throw std::runtime_error("Could not create OpenGL/OpenCL combo buffer " + std::to_string(openclError));

    for (auto& buffer : clBuffers)
    {
        if (buffer.first.empty())
            continue;
        buffer.second = std::shared_ptr<cl_mem>(new cl_mem(clCreateBuffer(*clContext, CL_MEM_READ_WRITE,
                width * height * 4 * sizeof(float), nullptr, &openclError)), DeleteReleaseMemObject);
        if (openclError)
		    throw std::runtime_error("Could not create OpenCL buffer " + std::to_string(openclError));
    }

	glBindTexture(GL_TEXTURE_2D, destructor->glTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    kernel->SetLaunchSize(width, height);

    if (clEnqueueAcquireGLObjects(*kernel->GetQueue(), 1, clBuffers[""].get(), 0, nullptr, nullptr))
		throw std::runtime_error("Could not acquire GL objects");

	HandleErr(glGetError());
}

void ClamInterop::MkBuffer(std::string buffername)
{
    cl_int openclError;
    clBuffers[buffername] = std::shared_ptr<cl_mem>(new cl_mem(clCreateBuffer(*clContext, CL_MEM_READ_WRITE,
            width * height * 4 * sizeof(float), nullptr, &openclError)), DeleteReleaseMemObject);
    if (openclError)
	    throw std::runtime_error("Could not create OpenCL buffer " + std::to_string(openclError));
}

std::shared_ptr<cl_mem> ClamInterop::GetBuffer(std::string buffername)
{
    if (clBuffers.find(buffername) == clBuffers.end())
        throw std::runtime_error("Buffer " + buffername + " did not exist");
    return clBuffers[buffername];
}

void ClamInterop::Blit(cl_command_queue const& queue)
{
    if (clEnqueueReleaseGLObjects(queue, 1, clBuffers[""].get(), 0, nullptr, nullptr))
		throw std::runtime_error("Could not release GL objects from OpenCL");

	if (clFinish(queue))
		throw std::runtime_error("Could not finish OpenCL (did kernel crash?)");

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, destructor->glBuffer);
	glBindTexture(GL_TEXTURE_2D, destructor->glTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, nullptr);

	glBegin(GL_QUADS);
	glTexCoord2f(0.f, 1.f);
	glVertex3f(0.f, 0.f, 0.f);
	glTexCoord2f(0.f, 0.f);
	glVertex3f(0.f, 1.f, 0.f);
	glTexCoord2f(1.f, 0.f);
	glVertex3f(1.f, 1.f, 0.f);
	glTexCoord2f(1.f, 1.f);
	glVertex3f(1.f, 0.f, 0.f);
	glEnd();

	HandleErr(glGetError());

    // -- //

    glFinish();

	if (clEnqueueAcquireGLObjects(queue, 1, clBuffers[""].get(), 0, nullptr, nullptr))
		throw std::runtime_error("Could not acquire GL objects");
}
