#include "display.h"
#include "option.h"

struct SdlInitClass
{
    SdlInitClass()
    {
        if (SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO))
        {
            throw std::runtime_error(SDL_GetError());
        }
        if (TTF_Init())
        {
            throw std::runtime_error(TTF_GetError());
        }
    }

    ~SdlInitClass()
    {
        TTF_Quit();
        SDL_Quit();
    }
} sdlInitInstance;

DisplayWindow::DisplayWindow(int x, int y, int width, int height) : fpsAverage(0), timeSinceLastTitle(0)
{
    isUserInput = IsUserInput();
    // TODO: SDL_WINDOW_FULLSCREEN
    Uint32 flags;
    flags = SDL_WINDOW_RESIZABLE;
    window = SDL_CreateWindow("Clam3", x, y, width, height, flags);
    std::string fontname = FontName();
    if (fontname.empty())
    {
        fontname = "/usr/share/fonts/TTF/Inconsolata-Regular.ttf";
    }
    if (IsUserInput())
    {
        font = TTF_OpenFont(fontname.c_str(), 14);
        if (!font)
        {
            throw std::runtime_error("Could not open font " + fontname + " (specify with $CLAM3_FONT)");
        }
    }
    else
    {
        font = NULL;
    }
}

DisplayWindow::~DisplayWindow()
{
    if (font)
    {
        TTF_CloseFont(font);
    }
    if (window)
    {
        SDL_DestroyWindow(window);
    }
}

bool DisplayWindow::UserInput(Kernel *kernel, double timePassed)
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        if (isUserInput)
        {
            kernel->UserInput(event);
        }
        if (event.type == SDL_QUIT)
        {
            return false;
        }
    }
    double weight = 1 / (fpsAverage + 1);
    fpsAverage = (timePassed + fpsAverage * weight) / (weight + 1);
    timeSinceLastTitle += timePassed;
    while (timeSinceLastTitle > 1.0)
    {
        SDL_Delay(1);
        SDL_SetWindowTitle(window, ("Clam3 - " + tostring(1 / fpsAverage) + " fps").c_str());
        timeSinceLastTitle--;
    }
    if (isUserInput)
    {
        kernel->Integrate(timePassed);
    }
    return true;
}
