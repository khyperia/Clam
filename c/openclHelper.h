#pragma once
#include <CL/cl.h>
#include "glclContext.h"

int newClContext(struct ClContext* result,
        struct Interop interop, char** sources, int sourcesLength);
void deleteClContext(struct ClContext context);
int setKernelArg(struct ClContext* context, const char* kernel,
        cl_uint index, void* arg, size_t argsize);
int invokeKernel(struct ClContext* context, struct Interop interop,
        const char* kernel, size_t invokeSize[2]);
