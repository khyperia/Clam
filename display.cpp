#include "display.h"
#include "option.h"

DisplayWindow::DisplayWindow(int x, int y, int width, int height)
{
    bool isCompute = IsCompute();
    isUserInput = IsUserInput();
    static bool init = false;
    if (!init)
    {
        SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO);
        init = true;
    }
    // TODO: SDL_WINDOW_FULLSCREEN
    Uint32 flags;
    flags = SDL_WINDOW_RESIZABLE;
    if (isCompute)
    {
        flags |= SDL_WINDOW_OPENGL;
    }
    window = SDL_CreateWindow("Clam3", x, y, width, height, flags);
    context = NULL;
    if (isCompute)
    {
        //context = SDL_GL_CreateContext(window);
    }
    else
    {
        //context = NULL;
    }
    lastTicks = SDL_GetTicks();
}

DisplayWindow::~DisplayWindow()
{
    if (context)
        SDL_GL_DeleteContext(context);
    if (window)
        SDL_DestroyWindow(window);
    SDL_Quit();
}

bool DisplayWindow::UserInput(Kernel* kernel)
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
    Uint32 newTicks = SDL_GetTicks();
    double time = (newTicks - lastTicks) / 1000.0;
    lastTicks = newTicks;
    if (isUserInput)
    {
        kernel->Integrate(time);
    }
    return true;
}
