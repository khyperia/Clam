#pragma once

class Kernel;

#include "network.h"
#include <SDL.h>
#include <SDL_net.h>
#include <set>
#include <CL/cl.hpp>

class Kernel
{
    std::set<SDL_Keycode> pressedKeys;

    virtual void OneTimeKeypress(SDL_Keycode keycode) = 0;

    virtual void RepeatKeypress(SDL_Keycode keycode, double time) = 0;

protected:
    void Serialize();

    void TryDeserialize();

public:
    cl::CommandQueue queue;
    cl::Context context;

    virtual ~Kernel()
    { };

    virtual void SendState(StateSync *output) const = 0;

    virtual void RecvState(StateSync *input) = 0;

    virtual void SaveWholeState(StateSync *output) const = 0;

    virtual void LoadWholeState(StateSync *input) = 0;

    virtual void RenderInto(cl::Memory memory, size_t width, size_t height) = 0;

    virtual std::string Name() = 0;

    void UserInput(SDL_Event event);

    void Integrate(double time);
};

Kernel *MakeKernel();
