#include "interop.h"
#include "helper.h"

ClamInterop::ClamInterop()
{
}

    ClamInterop::ClamInterop(std::shared_ptr<cl::Context> context)
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
    clBuffer = std::make_shared<cl::BufferGL>(*clContext, CL_MEM_WRITE_ONLY, destructor->glBuffer);

    glBindTexture(GL_TEXTURE_2D, destructor->glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, 4, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

const std::vector<cl::Memory> ClamInterop::RenderPre(ClamKernel& kernel)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFinish();
    std::vector<cl::Memory> vec;
    vec.push_back(*clBuffer);
    HandleErr(kernel.GetQueue()->enqueueAcquireGLObjects(&vec));

    kernel.SetLaunchSize(cl::NDRange(width, height));

    return vec;
}

void ClamInterop::RenderPost(ClamKernel& kernel, const std::vector<cl::Memory>& pre)
{
    HandleErr(kernel.GetQueue()->enqueueReleaseGLObjects(&pre));
    kernel.GetQueue()->finish();

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
