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
    virtual std::string Describe(const SettingCollection &settings) const;
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
    std::string pos_name;
    std::string move_speed;
    SDL_Scancode up;
    SDL_Scancode down;
    SDL_Scancode left;
    SDL_Scancode right;

public:
    Pan2d(std::string pos_name,
          std::string move_speed,
          SDL_Scancode up,
          SDL_Scancode down,
          SDL_Scancode left,
          SDL_Scancode right);
    ~Pan2d() override;

    void Integrate(SettingCollection &settings, double time) override;
};

class Camera3d: public UiSetting
{
    std::string pos_name;
    std::string look_name;
    std::string up_name;
    std::string fov;
    std::string move_speed;
    SDL_Scancode forwards;
    SDL_Scancode back;
    SDL_Scancode up;
    SDL_Scancode down;
    SDL_Scancode left;
    SDL_Scancode right;
    SDL_Scancode pitch_up;
    SDL_Scancode pitch_down;
    SDL_Scancode yaw_left;
    SDL_Scancode yaw_right;
    SDL_Scancode roll_left;
    SDL_Scancode roll_right;

public:
    Camera3d(std::string pos_name,
             std::string look_name,
             std::string up_name,
             std::string fov,
             std::string move_speed,
             SDL_Scancode forwards,
             SDL_Scancode back,
             SDL_Scancode up,
             SDL_Scancode down,
             SDL_Scancode left,
             SDL_Scancode right,
             SDL_Scancode pitch_up,
             SDL_Scancode pitch_down,
             SDL_Scancode yaw_left,
             SDL_Scancode yaw_right,
             SDL_Scancode roll_left,
             SDL_Scancode roll_right);
    ~Camera3d() override;

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

class InteractiveSetting: public UiSetting
{
    std::vector<std::pair<std::string, double>> settings;
    SDL_Scancode up;
    SDL_Scancode down;
    SDL_Scancode increase;
    SDL_Scancode decrease;
    int currentIndex;
    std::pair<std::string, double> &
    Current(const SettingCollection &settings, int *elem);
    int Size(const SettingCollection &settings);
public:
    InteractiveSetting(std::vector<std::pair<std::string, double>> settings,
                       SDL_Scancode up,
                       SDL_Scancode down,
                       SDL_Scancode increase,
                       SDL_Scancode decrease);
    ~InteractiveSetting() override;

    void Input(SettingCollection &settings, SDL_Event event) override;
    void Integrate(SettingCollection &settings, double time) override;
    std::string Describe(const SettingCollection &settings) const override;
};