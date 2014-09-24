#pragma once

#include <map>

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

    std::map<std::string, std::shared_ptr<cl_mem>> clBuffers;
    std::shared_ptr<cl_context> clContext;
    std::shared_ptr<Destructor> destructor;
    //void RenderPre(ClamKernel& kernel);
	//void RenderPost(ClamKernel& kernel);
    public:
    int width, height;
    ClamInterop();
    ClamInterop(std::shared_ptr<cl_context> context);
    void Resize(cl_command_queue const& queue, int width, int height);
    void MkBuffer(std::string buffername);
    std::shared_ptr<cl_mem> GetBuffer(std::string buffername);
    void Blit(cl_command_queue const& queue);
    /*
    template<typename... Args>
        void Render(ClamKernel kernel, Args... args)
        {
			RenderPre(kernel);
            kernel.Run(*clBuffers[""], width, height, args...);
            RenderPost(kernel);
        }
        */
};
