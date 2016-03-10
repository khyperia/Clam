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

    // byContext<allModules<setting> >
    std::vector<std::vector<SettingModuleBase *> > settings;
    std::vector<std::vector<KernelModuleBase *> > modules;

    std::vector<CUcontext> contexts;
    std::vector<CUmodule> cuModules;
    std::vector<CUfunction> kernelMains;
    std::vector<CUstream> streams;

    Vector2<int> renderOffset;
    bool useRenderOffset;

    std::vector<int> frames;

    size_t maxLocalSize;

    void CommonOneTimeKeypress(SDL_Keycode keycode);

    SettingAnimation *animation;

    std::vector<CuMem<int> > gpuBuffers;
    std::vector<size_t> oldWidth;
    std::vector<size_t> oldHeight;

public:
    Kernel(std::string name, std::vector<CUcontext> contexts);

    ~Kernel();

    void SendState(StateSync *output, bool everything, int context) const;

    void RecvState(StateSync *input, bool everything, int context);

    void UpdateNoRender();

    void Resize(size_t width, size_t height, int context);

    void RenderInto(int *memory, size_t width, size_t height, int context);

    void LoadAnimation();

    void SetTime(double time, bool wrap, int context);

    void SetFramed(bool framed, int context);

    int GetFrame(int context);

    SDL_Surface *Configure(TTF_Font *font);

    std::string Name();

    void UserInput(SDL_Event event);

    void Integrate(double time);

    const CUstream& Stream(int context) const
    {
        return streams[context];
    }
};
