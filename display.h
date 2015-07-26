#pragma once

#include "kernel.h"
#include "network.h"
#include "vector.h"
#include <SDL.h>
#include <stdexcept>

struct RenderType;

class DisplayWindow
{
    SDL_GLContext context;
    Uint32 lastTicks;
    bool isUserInput;

    DisplayWindow(const DisplayWindow &)
    {
        throw std::runtime_error("DisplayWindow class should not be moved");
    }

public:
    SDL_Window *window;

    DisplayWindow(int x, int y, int width, int height);

    ~DisplayWindow();

    bool UserInput(Kernel* kernel);
};
