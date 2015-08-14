#pragma once

#include "display.h"
#include "kernel.h"

struct RenderType;

class Driver
{
    DisplayWindow *window;
    Kernel *kernel;
    RenderType *renderType;
    CUdevice cuDevice;
    CUcontext cuContext;
    Connection connection;
    Vector2<int> headlessWindowSize;
public:
    Driver();

    ~Driver();

    bool RunFrame();
};
