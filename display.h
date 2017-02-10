#pragma once

#include <string>
#include <SDL.h>
#include <SDL_ttf.h>
#include "movable.h"

struct BlitData
{
    int width, height;
    int32_t *data;
    BlitData(int32_t *data, int width, int height);
};

class RenderTarget : public Immobile
{
public:
    RenderTarget();
    virtual ~RenderTarget();
    RenderTarget(const RenderTarget &other) = delete;
    RenderTarget &operator=(const RenderTarget &other) = delete;
    virtual bool RequestSize(size_t *width, size_t *height) = 0;
    virtual void Blit(BlitData data, const std::string &text) = 0;
};

class SdlWindow : public RenderTarget
{
    SDL_Window *window;
    TTF_Font *font;
public:
    SdlWindow(int x, int y, int width, int height, const char *font);
    ~SdlWindow();
private:
    bool RequestSize(size_t *width, size_t *height) override;
    void Blit(BlitData data, const std::string &text) override;
};

class FileTarget : public RenderTarget
{
    std::string fileName;
public:
    static void Screenshot(std::string fileName, BlitData data);
    FileTarget(std::string fileName);
    ~FileTarget();
private:
    bool RequestSize(size_t *width, size_t *height) override;
    void Blit(BlitData data, const std::string &text) override;
};
