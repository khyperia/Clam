#pragma once

#include <map>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>

#include "kernel.h"
#include <memory>

class ClamInterop
{
    std::map<std::string, std::pair<std::shared_ptr<cl_mem>, long>> clBuffers;
    std::shared_ptr<cl_context> clContext;
    std::shared_ptr<GLuint> glTexture;
    std::shared_ptr<GLuint> glBuffer;
    std::shared_ptr<cl_command_queue> clCommandQueue;
    public:
    int width, height;
    ClamInterop();
    ClamInterop(std::shared_ptr<ClamContext> context);
    void Resize(int width, int height);
    void MkBuffer(std::string buffername, long size); // -1 for dynamic screen size, SIZE IN BYTES
    void RmBuffer(std::string buffername);
    void DlBuffer(std::string buffername, long width); // Downloads and saves a buffer png to disk
    std::shared_ptr<cl_mem> GetBuffer(std::string buffername);
    void Blit();
    std::shared_ptr<cl_command_queue> GetQueue()
    {
        return clCommandQueue;
    }
};