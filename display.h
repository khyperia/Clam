#pragma once

#include "kernel.h"
#include "network.h"
#include "vector.h"
#include <SDL.h>
#include <SDL_ttf.h>
#include <stdexcept>

class DisplayWindow
{
    Uint32 lastTicks;
    bool isUserInput;

    DisplayWindow(const DisplayWindow &)
    {
        throw std::runtime_error("DisplayWindow class should not be moved");
    }

    DisplayWindow &operator=(const DisplayWindow &)
    {
        throw std::runtime_error("DisplayWindow class should not be moved");
    }

public:
    SDL_Window *window;
    TTF_Font *font;

    DisplayWindow(int x, int y, int width, int height);

    ~DisplayWindow();

    bool UserInput(Kernel *kernel);
};
