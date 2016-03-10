#pragma once

#include "display.h"
#include "kernel.h"

extern bool cudaSuccessfulInit;

struct RenderType;

struct BlitData
{
    int width, height;
    int32_t *data;
};

class Driver
{
    Kernel *kernel;
    RenderType *renderType;
    std::vector<CUcontext> cuContexts;
    Connection connection;
    Uint32 lastTickTime;
public:
    Driver();

    ~Driver();

    void MainLoop();
    void BlitImmediate(BlitData blitData, int context);
    void Tick();
};

void EnqueueCuMemFreeHost(void *hostPtr, int context);
void EnqueueBlitData(BlitData blitData, int context);
