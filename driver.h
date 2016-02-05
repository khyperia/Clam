#pragma once

#include "display.h"
#include "kernel.h"

extern bool cudaSuccessfulInit;

struct RenderType;

class Driver
{
    DisplayWindow *window;
    Kernel *kernel;
    RenderType *renderType;
    CUcontext cuContext;
    Connection connection;
    Vector2<int> headlessWindowSize;
    void UpdateWindowSize();
public:
    Driver();

    ~Driver();

    bool RunFrame(double timePassed);
};
