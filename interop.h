#pragma once

#define GL_GLEXT_PROTOTYPES
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
    const std::vector<cl::Memory> RenderPre(ClamKernel& kernel);
    void RenderPost(ClamKernel& kernel, const std::vector<cl::Memory>& pre);
    public:
    ClamInterop();
    ClamInterop(std::shared_ptr<cl::Context> context);
    void Resize(int width, int height);
    template<typename... Args>
        void Render(ClamKernel kernel, Args... args)
        {
            const std::vector<cl::Memory> vec = RenderPre(kernel);
            kernel.Run(vec[0], width, height, args...);
            RenderPost(kernel, vec);
        }
};
