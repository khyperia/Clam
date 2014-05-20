#pragma once

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>

#ifdef WIN32
#include <GL/glext.h>
//void(__stdcall *glGenBuffers)(GLsizei, GLuint*) = wglGetProcAddress("glGenBuffers");
#endif

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

    std::shared_ptr<cl_context> clContext;
    std::shared_ptr<Destructor> destructor;
    std::shared_ptr<cl_mem> clBuffer;
    int width, height;
    void RenderPre(ClamKernel& kernel);
	void RenderPost(ClamKernel& kernel);
    public:
    ClamInterop();
    ClamInterop(std::shared_ptr<cl_context> context);
    void Resize(int width, int height);
    template<typename... Args>
        void Render(ClamKernel kernel, Args... args)
        {
			RenderPre(kernel);
            kernel.Run(*clBuffer, width, height, args...);
            RenderPost(kernel);
        }
};
