#include "openclHelper.h"
#include "helper.h"
#include <stdio.h>
#include <string.h>

int newClContext(struct ClContext* result,
        struct Interop interop, char** sources, int sourcesLength)
{
    cl_int openclError = 0;
    result->program = clCreateProgramWithSource(interop.context,
            sourcesLength, (const char**)sources, NULL, &openclError);
    if (PrintErr(openclError && "clCreateProgramWithSource()"))
        return -1;
    openclError = clBuildProgram(result->program, 1, &interop.deviceId,
            "-cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -Werror", NULL, NULL);
    if (openclError)
    {
        size_t logSize;
        clGetProgramBuildInfo(result->program, interop.deviceId, CL_PROGRAM_BUILD_LOG,
                0, NULL, &logSize);
        
        char* str = (char*)malloc(logSize * sizeof(char));
        if (!str)
        {
            puts("malloc() failed. Exiting.");
            exit(-1);
        }
        clGetProgramBuildInfo(result->program, interop.deviceId, CL_PROGRAM_BUILD_LOG,
                logSize, str, NULL);
        printf("OpenCL build failed:\n%s\n", str);
        free(str);
        clReleaseProgram(result->program);
        return -1;
    }
    // kernels are created on-demand
    return 0;
}

void deleteClContext(struct ClContext context)
{
    struct cl_kernel_list* list = context.kernels;
    while (list)
    {
        free(list->name);
        clReleaseKernel(list->kernel);
        struct cl_kernel_list* temp = list->next;
        free(list);
        list = temp;
    }
    clReleaseProgram(context.program);
}

cl_kernel getKernelByName(struct ClContext* context, const char* name)
{
    struct cl_kernel_list** listRef = &context->kernels;
    while (*listRef)
    {
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
        *listRef = (struct cl_kernel_list*)malloc(sizeof(struct cl_kernel_list));
        (*listRef)->name = my_strdup(name);
        (*listRef)->kernel = kernel;
    }
    return (*listRef)->kernel;
}

int setKernelArg(struct ClContext* context, const char* kernel,
        cl_uint index, void* arg, size_t argsize)
{
    cl_kernel kernelObj = getKernelByName(context, kernel);
    if (PrintErr(clSetKernelArg(kernelObj, index, argsize, arg)))
        return -1;
    return 0;
}

int invokeKernel(struct ClContext* context, struct Interop interop,
        const char* kernel, size_t invokeSize[2])
{
    cl_kernel kernelObj = getKernelByName(context, kernel);
    size_t local[] = { 8, 8 };
    size_t global[] = {
        (invokeSize[0] + local[0] - 1) / local[0] * local[0],
        (invokeSize[1] + local[1] - 1) / local[1] * local[1]
    };
    if (PrintErr(clEnqueueNDRangeKernel(interop.command_queue, kernelObj, 2, NULL, global, local,
                    0, NULL, NULL)))
        return -1;
    return 0;
}
