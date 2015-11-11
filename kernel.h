#pragma once

class Kernel;

#include "network.h"
#include "cumem.h"
#include "vector.h"
#include <SDL.h>
#include <SDL_net.h>
#include <SDL_ttf.h>
#include <set>
#include <cuda.h>

class SettingModuleBase;

class KernelModuleBase;

class SettingAnimation;

class Kernel
{
    std::string name;

    std::set<SDL_Keycode> pressedKeys;

    std::vector<SettingModuleBase *> settings;
    std::vector<KernelModuleBase *> modules;

    CUmodule cuModule;
    CUfunction kernelMain;

    Vector2<int> renderOffset;
    bool useRenderOffset;

    int frame;

    size_t maxLocalSize;

    void CommonOneTimeKeypress(SDL_Keycode keycode);

    SettingAnimation *animation;

    CuMem<int> gpuBuffer;
    size_t oldWidth;
    size_t oldHeight;

public:
    Kernel(std::string name);

    ~Kernel();

    void SendState(StateSync *output, bool everything) const;

    void RecvState(StateSync *input, bool everything);

    void UpdateNoRender();

    void RenderInto(int *memory, size_t width, size_t height);

    void LoadAnimation();

    void SetTime(double time, bool wrap);

    void SetFramed(bool framed);

    SDL_Surface *Configure(TTF_Font *font);

    std::string Name();

    void UserInput(SDL_Event event);

    void Integrate(double time);
};
