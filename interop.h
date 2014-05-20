#pragma once

#include <GL/glew.h>
#include "context.h"
#include "kernel.h"
#include <memory>

class ClamInterop
{
    struct Destructor
    {
        ~Destructor();
        GLuint glTexture;
        GLuint glBuffer;
    };

    std::shared_ptr<cl::Context> clContext;
    std::shared_ptr<Destructor> destructor;
    std::shared_ptr<cl::BufferGL> clBuffer;
    int width, height;
    public:
    ClamInterop();
    ClamInterop(std::shared_ptr<cl::Context> context);
    void Resize(int width, int height);
    template<typename... Args>
        void Render(ClamKernel kernel, Args... args)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glFinish();
            std::vector<cl::Memory> vec;
            vec.push_back(*clBuffer);
            HandleErr(kernel.GetQueue()->enqueueAcquireGLObjects(&vec));

            kernel.SetLaunchSize(cl::NDRange(width, height));
            kernel.Run(vec[0], width, height, args...);
            
            HandleErr(kernel.GetQueue()->enqueueReleaseGLObjects(&vec));
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
};
