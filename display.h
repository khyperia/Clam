#pragma once

#include <SDL.h>
#include <SDL_ttf.h>

void WindowCreate(SDL_Window **window, TTF_Font **font, int x, int y, int width, int height);
void WindowDestroy(SDL_Window *window, TTF_Font *font);
