#pragma once

#include "kernelSetting.h"
#include <SDL_events.h>
#include <unordered_set>

class UiSetting: public Immobile
{
    std::unordered_set<SDL_Scancode> pressed_keys;
protected:
    bool IsKey(SDL_Scancode code);
public:
    UiSetting();
    virtual ~UiSetting();
    virtual void Integrate(SettingCollection &settings, double time);
    virtual void Input(SettingCollection &settings, SDL_Event event);
};

class ExpVar: public UiSetting
{
    std::string name;
    double rate;
    SDL_Scancode increase;
    SDL_Scancode decrease;

public:
    ExpVar(std::string name,
           double rate,
           SDL_Scancode increase,
           SDL_Scancode decrease);
    ~ExpVar() override;

    void Integrate(SettingCollection &settings, double time) override;
};

class Pan2d: public UiSetting
{
    std::string pos_x;
    std::string pos_y;
    std::string move_speed;
    SDL_Scancode up;
    SDL_Scancode down;
    SDL_Scancode left;
    SDL_Scancode right;

public:
    Pan2d(std::string pos_x,
          std::string pos_y,
          std::string move_speed,
          SDL_Scancode up,
          SDL_Scancode down,
          SDL_Scancode left,
          SDL_Scancode right);
    ~Pan2d() override;

    void Integrate(SettingCollection &settings, double time) override;
};

class Toggle: public UiSetting
{
    std::string name;
    SDL_Scancode key;

public:
    Toggle(std::string name, SDL_Scancode key);
    ~Toggle() override;

    void Input(SettingCollection &settings, SDL_Event event) override;
};