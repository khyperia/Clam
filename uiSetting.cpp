#include "uiSetting.h"

UiSetting::UiSetting()
    : pressed_keys()
{
}

UiSetting::~UiSetting()
{
}

void UiSetting::Integrate(SettingCollection &, double)
{
}

void UiSetting::Input(SettingCollection &, SDL_Event event)
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

bool UiSetting::IsKey(SDL_Scancode code)
{
    return pressed_keys.count(code) > 0;
}

ExpVar::ExpVar(std::string name,
               double rate,
               SDL_Scancode increase,
               SDL_Scancode decrease)
    : name(std::move(name)), rate(rate), increase(increase), decrease(decrease)
{
}

ExpVar::~ExpVar()
{
}

void ExpVar::Integrate(SettingCollection &settings, double time)
{
    auto &value = settings.Get(this->name).AsFloat();
    if (IsKey(increase))
    {
        value *= exp(rate * time);
    }
    if (IsKey(decrease))
    {
        value *= exp(-rate * time);
    }
}

Pan2d::Pan2d(std::string pos_x,
             std::string pos_y,
             std::string move_speed,
             SDL_Scancode up,
             SDL_Scancode down,
             SDL_Scancode left,
             SDL_Scancode right)
    : UiSetting(), pos_x(std::move(pos_x)), pos_y(std::move(pos_y)),
      move_speed(std::move(move_speed)), up(up), down(down), left(left),
      right(right)
{
}

Pan2d::~Pan2d()
{
}

void Pan2d::Integrate(SettingCollection &settings, double time)
{
    UiSetting::Integrate(settings, time);
    const auto pan_speed = 1.0f / 4.0f;
    auto &posx = settings.Get(this->pos_x).AsFloat();
    auto &posy = settings.Get(this->pos_y).AsFloat();
    auto &move_speed = settings.Get(this->move_speed).AsFloat();
    if (IsKey(up))
    {
        posy -= move_speed * pan_speed * time;
    }
    if (IsKey(down))
    {
        posy += move_speed * pan_speed * time;
    }
    if (IsKey(left))
    {
        posx -= move_speed * pan_speed * time;
    }
    if (IsKey(right))
    {
        posx += move_speed * pan_speed * time;
    }
}

Toggle::Toggle(std::string name, SDL_Scancode key)
    : UiSetting(), name(std::move(name)), key(std::move(key))
{
}

Toggle::~Toggle()
{
}

void Toggle::Input(SettingCollection &settings, SDL_Event event)
{
    UiSetting::Input(settings, event);
    if (event.type == SDL_KEYDOWN && event.key.keysym.scancode == key)
    {
        auto &value = settings.Get(this->name).AsBool();
        value = !value;
    }
}
