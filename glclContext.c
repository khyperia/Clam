#include "glclContext.h"
#include "openclHelper.h"
#include "helper.h"
#include <limits.h>
#include <GL/glx.h>
#include <stdio.h>
#include <string.h>

cl_context getClContext(cl_device_id* outputClDeviceId)
{
    cl_uint openclPlatformCount = 0;
    if (PrintErr(clGetPlatformIDs(0, NULL, &openclPlatformCount)))
        return NULL;

    cl_platform_id platformIds[openclPlatformCount];
    if (PrintErr(clGetPlatformIDs(openclPlatformCount, platformIds, NULL)))
        return NULL;

    cl_context context = NULL;

    for (cl_uint platform = 0; platform < openclPlatformCount; platform++)
    {
        cl_platform_id platformId = platformIds[platform];

        cl_uint openclDeviceCount = 0;
        if (PrintErr(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU,
                        0, NULL, &openclDeviceCount)))
            return NULL;

        cl_device_id deviceIds[openclDeviceCount];
        if (PrintErr(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, openclDeviceCount,
                        deviceIds, NULL)))
            return NULL;

        cl_context_properties contextProperties[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)(void*)glXGetCurrentContext(),
            CL_GLX_DISPLAY_KHR, (cl_context_properties)(void*)glXGetCurrentDisplay(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)platformId,
            0, 0
        };

        for (cl_uint device = 0; device < openclDeviceCount; device++)
        {
            cl_device_id deviceId = deviceIds[device];

            cl_int clError;
            context = clCreateContext(contextProperties,
                    1, &deviceId, 0, 0, &clError);
            if (clError == CL_SUCCESS)
            {
                if (outputClDeviceId)
                    *outputClDeviceId = deviceId;
                return context;
            }
        }
    }
    return NULL;
}

// Returns the linked list node of that key, or null if not found
struct cl_mem_list* getMemList(struct Interop interop, int key)
{
    struct cl_mem_list* value = interop.clMems;
    while (value && value->key != key)
        value = value->next;
    return value;
}

// Returns the cl_mem of a specified key (and the size), or null if not found
cl_mem getMem(struct Interop interop, int key, size_t* memSize)
{
    struct cl_mem_list* memList = getMemList(interop, key);
    if (!memList)
        return NULL;
    if (memSize)
        *memSize = memList->memorySize;
    return memList->memory;
}

// Allocates and appends the pre-allocated cl_mem key/value pair to the list
int addMem(struct Interop* interop, int key, cl_mem mem, size_t memSize)
{
    struct cl_mem_list** toAssign = &interop->clMems;
    while (*toAssign)
    {
        if ((*toAssign)->key == key)
        {
            printf("Buffer ID %d already exists, cannot create it\n", key);
            return -1;
        }
        toAssign = &(*toAssign)->next;
    }
    *toAssign = malloc_s(sizeof(struct cl_mem_list));
    (*toAssign)->key = key;
    (*toAssign)->memory = mem;
    (*toAssign)->memorySize = memSize;
    (*toAssign)->next = NULL;
    return 0;
}

// Allocates a cl_mem object and calls addMem()
int allocMem(struct Interop* interop, int key, size_t memSize)
{
    cl_int openclError;
    cl_mem memory = clCreateBuffer(interop->context, CL_MEM_READ_WRITE, memSize,
        NULL, &openclError);
    if (PrintErr(openclError))
    {
        puts("  Was from clCreateBuffer()");
        return -1;
    }
    return addMem(interop, key, memory, memSize);
}

// Frees and deletes the key/value from the list
void freeMem(struct Interop* interop, int key)
{
    if (!interop->clMems)
        return;
    cl_mem toFree;
    if (interop->clMems->key == key)
    {
        struct cl_mem_list* temp = interop->clMems;
        interop->clMems = interop->clMems->next;
        toFree = temp->memory;
        free(temp);
    }
    else
    {
        struct cl_mem_list* prev = interop->clMems;
        while (prev && prev->next->key != key)
            prev = prev->next;
        if (!prev)
            return;
        struct cl_mem_list* temp = prev->next;
        prev->next = temp->next;
        toFree = temp->memory;
        free(temp);
    }
    clReleaseMemObject(toFree);
}

float* dlMem(struct Interop interop, int key, size_t* memSize, size_t screenSizeBytes)
{
    cl_mem clmem = getMem(interop, key, memSize);
    if (!clmem)
        return NULL;
    if (*memSize == 0)
        *memSize = screenSizeBytes;
    float* data = malloc_s(*memSize);
    if (PrintErr(clEnqueueReadBuffer(interop.command_queue, clmem, 1, 0,
                    *memSize, data, 0, NULL, NULL)))
    {
        free(data);
        return NULL;
    }
    return data;
}

// Creates an Interop struct
int newInterop(struct Interop* result)
{
    result->context = getClContext(&result->deviceId);
    result->clMems = NULL;
    memset(&result->clContext, 0, sizeof(struct ClContext));

    if (!result->context)
    {
        puts("No OpenCL devices found");
        return -1;
    }

    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, 1.f, 0.f, 1.f, -1.f, 1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glGenTextures(1, &result->glTexture);
    glGenBuffers(1, &result->glBuffer);

    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, result->glTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    if (PrintErr((int)glGetError()))
        return -1;

    cl_int openclError;
    result->command_queue = clCreateCommandQueue(result->context,
            result->deviceId, 0, &openclError);
    if (PrintErr(openclError))
        return -1;

    return 0;
}

// Cleans up an Interop struct (does not deallocate the cl_context)
void deleteInterop(struct Interop interop)
{
    if (interop.clContext.program)
        deleteClContext(interop.clContext);
    glDeleteTextures(1, &interop.glTexture);
    glDeleteBuffers(1, &interop.glBuffer);
    clReleaseCommandQueue(interop.command_queue);
    while (interop.clMems)
        freeMem(&interop, interop.clMems->key);
    clReleaseContext(interop.context);
}

// Handles a screen window resize, re-allocating affected buffers etc.
int resizeInterop(struct Interop* interop, int width, int height)
{
    // Bind the OpenGL buffers
    size_t bufferSize = (size_t)width * (size_t)height * 4 * sizeof(float);
    glViewport(0, 0, width, height);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, interop->glBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)bufferSize, NULL, GL_DYNAMIC_COPY);
    if (PrintErr((int)glGetError())) return -1;

    freeMem(interop, 0);
    cl_int openclError = 0;
    cl_mem openclBuffer = clCreateFromGLBuffer(interop->context, CL_MEM_READ_WRITE,
            interop->glBuffer, &openclError);
    if (PrintErr(openclError))
        return -1;
    if (PrintErr(clEnqueueAcquireGLObjects(interop->command_queue, 1,
                    &openclBuffer, 0, NULL, NULL)))
        return -1;
    if (PrintErr(addMem(interop, 0, openclBuffer, 0)))
        return -1;

    struct cl_mem_list* memList = interop->clMems;
    while (memList)
    {
        if (memList->key != 0 && memList->memorySize == (size_t)0)
        {
            clReleaseMemObject(memList->memory);
            memList->memory = clCreateBuffer(interop->context, CL_MEM_READ_WRITE, bufferSize,
                    NULL, &openclError);
            if (PrintErr(openclError && "clCreateBuffer()"))
                return -1;
        }
        memList = memList->next;
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, interop->glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    if (PrintErr((int)glGetError())) return -1;

    return 0;
}

// Displays buffer 0 onscreen (should have been created with resizeInterop)
int blitInterop(struct Interop interop, int width, int height)
{
    struct cl_mem_list* screenMemList = getMemList(interop, 0);
    if (!screenMemList)
    {
        // TODO: Consider calling resizeInterop here
        // This only happened ocasionally in the C++ version, though
        puts("Buffer 0 did not exist, cannot blit to screen");
        return -1;
    }
    cl_mem* screenMem = &screenMemList->memory;
    if (PrintErr(clEnqueueReleaseGLObjects(interop.command_queue, 1, screenMem, 0, NULL, NULL)))
        return -1;
    if (PrintErr(clFinish(interop.command_queue)))
        return -1;
    if (PrintErr((int)glGetError()))
        return -1;

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, interop.glBuffer);
    glBindTexture(GL_TEXTURE_2D, interop.glTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, NULL);
    if (PrintErr((int)glGetError())) return -1;

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
    glFinish();
    if (PrintErr((int)glGetError())) return -1;

    if (PrintErr(clEnqueueAcquireGLObjects(interop.command_queue, 1, screenMem, 0, NULL, NULL)))
        return -1;
    return 0;
}
