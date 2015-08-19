#pragma once

class Kernel;

#include "network.h"
#include "cumem.h"
#include <SDL.h>
#include <SDL_net.h>
#include <SDL_ttf.h>
#include <set>
#include <cuda.h>

class Kernel
{
    std::set<SDL_Keycode> pressedKeys;

    virtual void OneTimeKeypress(SDL_Keycode keycode) = 0;

    virtual void RepeatKeypress(SDL_Keycode keycode, double time) = 0;

public:
    virtual ~Kernel()
    { };

    virtual void SendState(StateSync *output) const = 0;

    virtual void RecvState(StateSync *input) = 0;

    virtual void SaveWholeState(StateSync *output) const = 0;

    virtual void LoadWholeState(StateSync *input) = 0;

    virtual void RenderInto(CuMem<int> &memory, size_t width, size_t height) = 0;

    virtual void SetTime(float time) = 0;

    virtual SDL_Surface *Configure(TTF_Font *font) = 0;

    virtual std::string Name() = 0;

    void UserInput(SDL_Event event);

    void Integrate(double time);
};

Kernel *MakeKernel();
