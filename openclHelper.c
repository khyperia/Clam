#include "openclHelper.h"
#include "helper.h"
#include <stdio.h>
#include <string.h>

// Builds an openCl kernel (fills in interop->clContext)
int newClContext(struct Interop* interop, char** sources, cl_uint sourcesLength)
{
    interop->clContext.kernels = NULL;
    cl_int openclError = 0;
    interop->clContext.program = clCreateProgramWithSource(interop->context,
            sourcesLength, (const char**)sources, NULL, &openclError);
    if (PrintErr(openclError && "clCreateProgramWithSource()"))
    {
        interop->clContext.program = NULL;
        return -1;
    }
    openclError = clBuildProgram(interop->clContext.program, 1, &interop->deviceId,
            "-cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -Werror", NULL, NULL);
    if (openclError)
    {
        size_t logSize;
        clGetProgramBuildInfo(interop->clContext.program, interop->deviceId,
                CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

        char* str = malloc_s(logSize * sizeof(char));
        clGetProgramBuildInfo(interop->clContext.program, interop->deviceId,
                CL_PROGRAM_BUILD_LOG, logSize, str, NULL);
        printf("OpenCL build failed:\n%s\n", str);
        free(str);
        clReleaseProgram(interop->clContext.program);
        interop->clContext.program = NULL;
        return -1;
    }
    // kernels are created on-demand
    return 0;
}

// Deletes/frees an openCl context
void deleteClContext(struct ClContext context)
{
    struct cl_kernel_list* list = context.kernels;
    while (list)
    {
        free(list->name);
        list->name = NULL;
        clReleaseKernel(list->kernel);
        list->kernel = NULL;
        struct cl_kernel_list* temp = list->next;
        list->next = NULL;
        free(list);
        list = temp;
    }
    clReleaseProgram(context.program);
}

// Retrieves a kernel by it's name, creating it if needed
cl_kernel getKernelByName(struct ClContext* context, const char* name)
{
    struct cl_kernel_list** listRef = &context->kernels;
    while (*listRef)
    {
        if (PrintErr(!(*listRef)->name))
            return NULL;
        if (strcmp((*listRef)->name, name) == 0)
            break;
        listRef = &(*listRef)->next;
    }
    if (!*listRef)
    {
        cl_int openclError = 0;
        cl_kernel kernel = clCreateKernel(context->program, name, &openclError);
        if (PrintErr(openclError /* clCreateKernel() */))
            return NULL;
        *listRef = malloc_s(sizeof(struct cl_kernel_list));
        (*listRef)->name = my_strdup(name);
        (*listRef)->kernel = kernel;
        (*listRef)->next = NULL;
    }
    return (*listRef)->kernel;
}

// Sets the specified kernel's arg
int setKernelArg(struct ClContext* context, const char* kernel,
        cl_uint index, void* arg, size_t argsize)
{
    cl_kernel kernelObj = getKernelByName(context, kernel);
    if (PrintErr(clSetKernelArg(kernelObj, index, argsize, arg)))
        return -1;
    return 0;
}

// Calls a kernel
int invokeKernel(struct ClContext* context, struct Interop interop,
        const char* kernel, size_t invokeSize[2])
{
    cl_kernel kernelObj = getKernelByName(context, kernel);
    size_t local[] = { 8, 8 };
    size_t global[] = {
        (invokeSize[0] + local[0] - 1) / local[0] * local[0],
        (invokeSize[1] + local[1] - 1) / local[1] * local[1]
    };
    if (PrintErr(clEnqueueNDRangeKernel(interop.command_queue, kernelObj,
                    2, NULL, global, local, 0, NULL, NULL)))
        return -1;
    return 0;
}
