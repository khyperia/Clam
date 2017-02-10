#include "lib_init.h"
#include <SDL.h>
#include <SDL_ttf.h>
#include <stdexcept>

static void InitSdl()
{
    if (SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO))
    {
        throw std::runtime_error("Error initializing SDL: " + std::string(SDL_GetError()));
    }
    if (TTF_Init())
    {
        throw std::runtime_error("Error initializing SDL_ttf: " + std::string(TTF_GetError()));
    }
}

static void DeInitSdl()
{
    TTF_Quit();
    SDL_Quit();
}

static int num_windows = 0;

void IncrementSdlUsage()
{
    if (num_windows == 0)
    {
        InitSdl();
    }
    num_windows++;
}

void DecrementSdlUsage()
{
    num_windows--;
    if (num_windows == 0)
    {
        DeInitSdl();
    }
}
