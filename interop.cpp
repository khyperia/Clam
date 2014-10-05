#include "interop.h"
#include "helper.h"
#include "image.h"
#include <CL/cl_gl.h>

ClamInterop::ClamInterop()
{
}

ClamInterop::ClamInterop(std::shared_ptr<ClamContext> context)
    : clContext(context->GetContext())
{
    glClearColor(1.f, 0.f, 0.f, 1.f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, 1.f, 0.f, 1.f, -1.f, 1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    GLuint aglTexture, aglBuffer;

    glGenTextures(1, &aglTexture);
    glGenBuffers(1, &aglBuffer);

    glTexture = make_custom_shared<GLuint>([](GLuint const& dying)
            {
                glDeleteTextures(1, &dying);
            }, aglTexture);
    glBuffer = make_custom_shared<GLuint>([](GLuint const& dying)
            {
                glDeleteBuffers(1, &dying);
            }, aglBuffer);

    glEnable(GL_TEXTURE_2D);
    
    glBindTexture(GL_TEXTURE_2D, *glTexture);
    HandleErr(glGetError());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    HandleErr(glGetError());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    HandleErr(glGetError());
    
    cl_int openclError = 0;
    cl_command_queue openclCommandQueue =
        clCreateCommandQueue(*context->GetContext(), *context->GetDevice(), 0, &openclError);
    if (openclError)
        throw std::runtime_error("Failed to create command queue");

    clCommandQueue = make_custom_shared<cl_command_queue>([](cl_command_queue const& dying)
            {
                clReleaseCommandQueue(dying);
            }, openclCommandQueue);
}

void ClamInterop::Resize(int _width, int _height)
{
    width = _width;
    height = _height;
    HandleErr(glGetError());
    glViewport(0, 0, width, height);
    HandleErr(glGetError());
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *glBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
            static_cast<GLsizeiptr>(width) * static_cast<GLsizeiptr>(height) *
                4 * static_cast<GLsizeiptr>(sizeof(float)),
            nullptr, GL_DYNAMIC_COPY);
    HandleErr(glGetError());
    
    cl_int openclError = 0;
    cl_mem openclBuffer = clCreateFromGLBuffer(*clContext,
            CL_MEM_WRITE_ONLY, *glBuffer, &openclError);
    clBuffers[""] = std::make_pair(make_custom_shared<cl_mem>([](cl_mem const& dying)
            {
                clReleaseMemObject(dying);
            }, openclBuffer), -1);
    
    if (openclError)
        throw std::runtime_error("Could not create OpenGL/OpenCL combo buffer "
                + std::to_string(openclError));

    for (auto& buffer : clBuffers)
    {
        if (buffer.first.empty() || buffer.second.second != -1)
            continue;
        auto temp = clCreateBuffer(*clContext, CL_MEM_READ_WRITE,
                static_cast<size_t>(width) * static_cast<size_t>(height) *
                    4 * static_cast<size_t>(sizeof(float)),
                nullptr, &openclError);
        if (openclError)
            throw std::runtime_error("Could not create OpenCL buffer "
                    + std::to_string(openclError));
        buffer.second.first = make_custom_shared<cl_mem>([](cl_mem const& dying)
                {
                    clReleaseMemObject(dying);
                }, temp);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // Don't ask why this is needed.
    glBindTexture(GL_TEXTURE_2D, *glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    HandleErr(glGetError());

    if (clEnqueueAcquireGLObjects(*clCommandQueue, 1,
                GetBuffer("").get(), 0, nullptr, nullptr))
        throw std::runtime_error("Could not acquire GL objects");

    HandleErr(glGetError());
}

void ClamInterop::MkBuffer(std::string buffername, long size)
{
    cl_int openclError;
    size_t actualsize = size == -1 ? static_cast<size_t>(width) * static_cast<size_t>(height) *
        4 * static_cast<size_t>(sizeof(float)) : static_cast<size_t>(size);
    auto temp = clCreateBuffer(*clContext, CL_MEM_READ_WRITE, actualsize, nullptr, &openclError);
    if (openclError)
        throw std::runtime_error("Could not create OpenCL buffer "
                + std::to_string(openclError));
    clBuffers[buffername] = std::make_pair(make_custom_shared<cl_mem>([](cl_mem const& dying)
            {
                clReleaseMemObject(dying);
            }, temp), size);
}

void ClamInterop::RmBuffer(std::string buffername)
{
    auto found = clBuffers.find(buffername);
    if (found != clBuffers.end())
        clBuffers.erase(found);
    else
        throw std::runtime_error("Buffer \"" + buffername + "\" not found");
}

void ClamInterop::DlBuffer(std::string buffername, long imagewidth)
{
    auto buffer = clBuffers[buffername];
    std::vector<float> cpuData(static_cast<size_t>(buffer.second) / sizeof(float));
    clEnqueueReadBuffer(*clCommandQueue, *buffer.first, true, 0,
            static_cast<size_t>(buffer.second),
            cpuData.data(), 0, nullptr, nullptr);
    clFinish(*clCommandQueue);
    WriteImage(buffername + ".png", cpuData, static_cast<unsigned long>(imagewidth));
}

std::shared_ptr<cl_mem> ClamInterop::GetBuffer(std::string buffername)
{
    if (clBuffers.find(buffername) == clBuffers.end())
        throw std::runtime_error("Buffer " + buffername + " did not exist");
    return clBuffers[buffername].first;
}

void ClamInterop::Blit()
{
    if (clEnqueueReleaseGLObjects(*clCommandQueue, 1, GetBuffer("").get(), 0, nullptr, nullptr))
        throw std::runtime_error("Could not release GL objects from OpenCL");

    int error;
    if ((error = clFinish(*clCommandQueue)))
        throw std::runtime_error("Could not finish OpenCL (did kernel crash?) Error = "
                + std::to_string(error));

    HandleErr(glGetError());

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *glBuffer);
    glBindTexture(GL_TEXTURE_2D, *glTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, nullptr);
    HandleErr(glGetError());

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

    glFinish();

    if (clEnqueueAcquireGLObjects(*clCommandQueue, 1, GetBuffer("").get(), 0, nullptr, nullptr))
        throw std::runtime_error("Could not acquire GL objects");
    HandleErr(glGetError());
}
