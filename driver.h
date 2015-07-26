#pragma once

#include "display.h"
#include "kernel.h"

class Driver
{
    DisplayWindow *window;
    Kernel *kernel;
    RenderType *renderType;
    Connection connection;
    Vector2<int> headlessWindowSize;
public:
    Driver();

    ~Driver();

    bool RunFrame();
};
