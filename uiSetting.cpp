#include "uiSetting.h"

UiSetting::~UiSetting()
{
}

Camera2d::Camera2d()
    : pressed_keys()
{
}

Camera2d::~Camera2d()
{
}

void Camera2d::Integrate(SettingCollection &settings, float time)
{
    const auto zoom_speed = 1.0f;
    const auto pan_speed = 1.0f / 4.0f;
    auto &posx = settings.Get("posx").AsFloat();
    auto &posy = settings.Get("posy").AsFloat();
    auto &zoom = settings.Get("zoom").AsFloat();
    if (pressed_keys.count(SDL_SCANCODE_W) > 0)
    {
        posy -= zoom * pan_speed * time;
    }
    if (pressed_keys.count(SDL_SCANCODE_S) > 0)
    {
        posy += zoom * pan_speed * time;
    }
    if (pressed_keys.count(SDL_SCANCODE_A) > 0)
    {
        posx -= zoom * pan_speed * time;
    }
    if (pressed_keys.count(SDL_SCANCODE_D) > 0)
    {
        posx += zoom * pan_speed * time;
    }
    if (pressed_keys.count(SDL_SCANCODE_R) > 0)
    {
        zoom *= exp(-zoom_speed * time);
    }
    if (pressed_keys.count(SDL_SCANCODE_F) > 0)
    {
        zoom *= exp(zoom_speed * time);
    }
}

void Camera2d::Input(SettingCollection &, SDL_Event event)
{
    if (event.type == SDL_KEYDOWN)
    {
        pressed_keys.insert(event.key.keysym.scancode);
    }
    else if (event.type == SDL_KEYUP)
    {
        pressed_keys.erase(event.key.keysym.scancode);
    }
}
