#include "display.h"
#include "lib_init.h"
#include <iostream>
#include "util.h"

BlitData::BlitData(int32_t *data, int width, int height)
    : width(width), height(height), data(data)
{
}

RenderTarget::RenderTarget()
{
}

RenderTarget::~RenderTarget()
{
}

SdlWindow::SdlWindow(int x, int y, int width, int height, const char *font)
{
    IncrementSdlUsage();
    Uint32 flags = SDL_WINDOW_RESIZABLE;
    // TODO: SDL_WINDOW_FULLSCREEN
    this->window = SDL_CreateWindow("Clam3", x, y, width, height, flags);
    if (font != NULL)
    {
        if (font[0] == '\0')
        {
            font = "/usr/share/fonts/TTF/Inconsolata-Regular.ttf";
        }
        this->font = TTF_OpenFont(font, 14);
        if (!this->font)
        {
            throw std::runtime_error("Could not open font " + std::string(font)
                                         + " (specify with $CLAM3_FONT)");
        }
    }
    else
    {
        this->font = NULL;
    }
}

SdlWindow::~SdlWindow()
{
    if (font)
    {
        TTF_CloseFont(font);
        font = NULL;
    }
    if (window)
    {
        SDL_DestroyWindow(window);
        window = NULL;
    }
    DecrementSdlUsage();
}

bool SdlWindow::RequestSize(size_t *width, size_t *height)
{
    int width_int, height_int;
    SDL_GetWindowSize(window, &width_int, &height_int);
    if (width)
    {
        *width = (size_t)width_int;
    }
    if (height)
    {
        *height = (size_t)height_int;
    }
    return true;
}

void SdlWindow::Blit(BlitData data, const std::string &text)
{
    SDL_Surface *surface = SDL_GetWindowSurface(window);
    if (SDL_LockSurface(surface))
    {
        throw std::runtime_error("Could not lock SDL surface");
    }
    if (surface->format->BytesPerPixel != 4)
    {
        throw std::runtime_error("Window surface bytes/pixel != 4");
    }
    size_t pixbuf_size = (size_t)data.width * data.height;
    size_t surface_size = (size_t)surface->w * surface->h;
    size_t size = pixbuf_size < surface_size ? pixbuf_size : surface_size;
    size_t byte_size = size * surface->format->BytesPerPixel;
    memcpy(surface->pixels, data.data, byte_size);
    SDL_UnlockSurface(surface);
    if (!text.empty())
    {
        if (!this->font)
        {
            throw std::runtime_error(
                "Tried to render text, but no font was available");
        }
        SDL_Color color = {255, 0, 0, 0};
        SDL_Surface *surf = TTF_RenderText_Blended_Wrapped(font,
                                                           text.c_str(),
                                                           color,
                                                           (Uint32)data.width);
        SDL_BlitSurface(surf, NULL, surface, NULL);
        SDL_FreeSurface(surf);
    }
    SDL_UpdateWindowSurface(window);
}

FileTarget::FileTarget(std::string baseFileName)
    : baseFileName(baseFileName), curImage(0)
{
}

FileTarget::~FileTarget()
{
}

bool FileTarget::RequestSize(size_t *, size_t *)
{
    return false;
}

// Text parameter is ignored
void FileTarget::Blit(BlitData data, const std::string &)
{
    SDL_Surface *surface = SDL_CreateRGBSurface(0,
                                                data.width,
                                                data.height,
                                                4 * 8,
                                                255 << 16,
                                                255 << 8,
                                                255,
                                                0);
    if (SDL_LockSurface(surface))
    {
        throw std::runtime_error("Could not lock temp buffer surface");
    }
    memcpy(surface->pixels,
           data.data,
           (size_t)data.width * data.height * surface->format->BytesPerPixel);
    SDL_UnlockSurface(surface);

    std::string
        filename = this->baseFileName + "." + tostring(curImage) + ".bmp";
    int ret = SDL_SaveBMP(surface, filename.c_str());
    if (ret)
    {
        throw std::runtime_error(
            "Could not save image " + filename + ": " + SDL_GetError());
    }
    std::cout << "Saved image '" << filename << "'" << std::endl;

    SDL_FreeSurface(surface);
}
