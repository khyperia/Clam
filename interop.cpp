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

void ClamInterop::Resize(int _width, int _height)
{
	width = _width;
	height = _height;
	glViewport(0, 0, width, height);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, destructor->glBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(float), nullptr, GL_DYNAMIC_COPY);

	cl_int openclError = 0;
	cl_mem openclBuffer = clCreateFromGLTexture2D(*clContext, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, destructor->glBuffer, &openclError);
	clBuffer = std::make_shared<cl_mem>(openclBuffer);
	if (openclError)
		throw std::runtime_error("Could not create OpenGL/OpenCL combo buffer");

	glBindTexture(GL_TEXTURE_2D, destructor->glTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void ClamInterop::RenderPre(ClamKernel& kernel)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glFinish();

	if (clEnqueueAcquireGLObjects(*kernel.GetQueue(), 1, clBuffer.get(), 0, nullptr, nullptr))
		throw std::runtime_error("Could not acquire GL objects");

	kernel.SetLaunchSize(width, height);
}

void ClamInterop::RenderPost(ClamKernel& kernel)
{
	if (clEnqueueReleaseGLObjects(*kernel.GetQueue(), 1, clBuffer.get(), 0, nullptr, nullptr))
		throw std::runtime_error("Could not release GL objects");

	if (clFinish(*kernel.GetQueue()))
		throw std::runtime_error("Could not finish OpenCL kernel (did kernel crash?)");

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
}
