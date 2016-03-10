#include "display.h"
#include "option.h"
#include <iostream>

struct SdlInitClass
{
    SdlInitClass()
    {
        if (SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO))
        {
            std::cout << "Error initializing SDL, may fail later: " << SDL_GetError() << "\n";
        }
        if (TTF_Init())
        {
            std::cout << "Error initializing SDL_ttf, may fail later: " << TTF_GetError() << "\n";
        }
    }

    ~SdlInitClass()
    {
        TTF_Quit();
        SDL_Quit();
    }
} sdlInitInstance;

void WindowCreate(SDL_Window **window, TTF_Font **font, int x, int y, int width, int height)
{
    // TODO: SDL_WINDOW_FULLSCREEN
    Uint32 flags;
    flags = SDL_WINDOW_RESIZABLE;
    *window = SDL_CreateWindow("Clam3", x, y, width, height, flags);
    std::string fontname = FontName();
    if (fontname.empty())
    {
        fontname = "/usr/share/fonts/TTF/Inconsolata-Regular.ttf";
    }
    if (IsUserInput())
    {
        *font = TTF_OpenFont(fontname.c_str(), 14);
        if (!*font)
        {
            throw std::runtime_error("Could not open font " + fontname + " (specify with $CLAM3_FONT)");
        }
    }
    else
    {
        *font = NULL;
    }
}

void WindowDestroy(SDL_Window *window, TTF_Font *font)
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
