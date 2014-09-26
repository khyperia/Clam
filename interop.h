#pragma once

#include <map>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>

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
    public:
    int width, height;
    ClamInterop();
    ClamInterop(std::shared_ptr<cl_context> context);
    void Resize(std::shared_ptr<ClamKernel> queue, int width, int height);
    void MkBuffer(std::string buffername);
    std::shared_ptr<cl_mem> GetBuffer(std::string buffername);
    void Blit(cl_command_queue const& queue);
};
