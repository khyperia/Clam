#pragma once

#include "kernelSetting.h"
#include <SDL_events.h>
#include <unordered_set>

class UiSetting: public Immobile
{
public:
    virtual ~UiSetting();
    virtual void Integrate(SettingCollection &settings, float time) = 0;
    virtual void Input(SettingCollection &settings, SDL_Event event) = 0;
};

class Camera2d: public UiSetting
{
    std::unordered_set<SDL_Scancode> pressed_keys;

public:
    Camera2d();
    ~Camera2d() override;

    void Integrate(SettingCollection &settings, float time) override;
    void Input(SettingCollection &settings, SDL_Event event) override;
};