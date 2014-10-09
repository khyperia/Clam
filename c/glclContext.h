#pragma once
#define GL_GLEXT_PROTOTYPES
#include <CL/cl_gl.h>
#include <GL/gl.h>

// Begin should be in openclHelper.h
struct cl_kernel_list
{
    cl_kernel kernel;
    char* name;
    struct cl_kernel_list* next;
};

struct ClContext
{
    cl_program program;
    struct cl_kernel_list* kernels;
};
// End should be in openclHelper.h

struct cl_mem_list
{
    int key;
    cl_mem memory;
    size_t memorySize;
    struct cl_mem_list* next;
};

struct Interop
{
    struct ClContext clContext;
    struct cl_mem_list* clMems;
    cl_device_id deviceId;
    cl_context context;
    cl_command_queue command_queue;
    GLuint glTexture;
    GLuint glBuffer;
};

int newInterop(struct Interop* result);
void deleteInterop(struct Interop interop);
int resizeInterop(struct Interop* interop, int width, int height);
int blitInterop(struct Interop interop, int width, int height);
int allocMem(struct Interop* interop, int key, size_t memSize);
cl_mem getMem(struct Interop interop, int key, size_t* memSize);
void freeMem(struct Interop* interop, int key);
