#include "uiSetting.h"
#include "vector.h"

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

std::string UiSetting::Describe(const SettingCollection &) const
{
    return "";
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
    UiSetting::Integrate(settings, time);
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

Pan2d::Pan2d(std::string pos_name,
             std::string move_speed,
             SDL_Scancode up,
             SDL_Scancode down,
             SDL_Scancode left,
             SDL_Scancode right)
    : UiSetting(), pos_name(std::move(pos_name)),
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
    auto &pos = settings.Get(this->pos_name).AsVec2();
    auto &move_speed = settings.Get(this->move_speed).AsFloat();
    if (IsKey(up))
    {
        pos.y -= move_speed * pan_speed * time;
    }
    if (IsKey(down))
    {
        pos.y += move_speed * pan_speed * time;
    }
    if (IsKey(left))
    {
        pos.x -= move_speed * pan_speed * time;
    }
    if (IsKey(right))
    {
        pos.x += move_speed * pan_speed * time;
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

Camera3d::Camera3d(std::string pos_name,
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
                   SDL_Scancode roll_right)
    : pos_name(pos_name), look_name(look_name), up_name(up_name), fov(fov),
      move_speed(move_speed), forwards(forwards), back(back), up(up),
      down(down), left(left), right(right), pitch_up(pitch_up),
      pitch_down(pitch_down), yaw_left(yaw_left), yaw_right(yaw_right),
      roll_left(roll_left), roll_right(roll_right)
{
}

Camera3d::~Camera3d()
{
}

void Camera3d::Integrate(SettingCollection &settings, double time)
{
    UiSetting::Integrate(settings, time);
    auto &pos = settings.Get(this->pos_name).AsVec3();
    auto &look = settings.Get(this->look_name).AsVec3();
    auto &up = settings.Get(this->up_name).AsVec3();
    const auto move_speed = settings.Get(this->move_speed).AsFloat() * time;
    const auto turn_speed = settings.Get(this->fov).AsFloat() * time;
    if (IsKey(this->forwards))
    {
        pos = pos + look * move_speed;
    }
    if (IsKey(this->back))
    {
        pos = pos - look * move_speed;
    }
    if (IsKey(this->up))
    {
        pos = pos + up * move_speed;
    }
    if (IsKey(this->down))
    {
        pos = pos - up * move_speed;
    }
    if (IsKey(this->right))
    {
        pos = pos + cross(look, up) * move_speed;
    }
    if (IsKey(this->left))
    {
        pos = pos - cross(look, up) * move_speed;
    }
    if (IsKey(this->pitch_up))
    {
        look = look + up * turn_speed;
    }
    if (IsKey(this->pitch_down))
    {
        look = look - up * turn_speed;
    }
    if (IsKey(this->yaw_right))
    {
        look = look + cross(look, up) * turn_speed;
    }
    if (IsKey(this->yaw_left))
    {
        look = look - cross(look, up) * turn_speed;
    }
    if (IsKey(this->roll_right))
    {
        up = up + cross(look, up) * turn_speed;
    }
    if (IsKey(this->roll_left))
    {
        up = up - cross(look, up) * turn_speed;
    }
    look = look.normalized();
    up = cross(cross(look, up), look).normalized();
}

InteractiveSetting::InteractiveSetting(std::vector<std::pair<std::string,
                                                             double>> settings,
                                       SDL_Scancode up,
                                       SDL_Scancode down,
                                       SDL_Scancode increase,
                                       SDL_Scancode decrease)
    : settings(std::move(settings)), up(up), down(down), increase(increase),
      decrease(decrease), currentIndex(0)
{
}

InteractiveSetting::~InteractiveSetting()
{
}

std::pair<std::string, double> &
InteractiveSetting::Current(const SettingCollection &settings, int *elem)
{
    int cur = 0;
    for (auto &setting : this->settings)
    {
        const auto &val = settings.Get(setting.first);
        int delta = currentIndex - cur;
        if (val.IsVec2())
        {
            if (delta < 2)
            {
                if (elem)
                {
                    *elem = delta;
                }
                return setting;
            }
            cur += 2;
        }
        else if (val.IsVec3())
        {
            if (delta < 3)
            {
                if (elem)
                {
                    *elem = delta;
                }
                return setting;
            }
            cur += 3;
        }
        else
        {
            if (delta < 1)
            {
                if (elem)
                {
                    *elem = delta;
                }
                return setting;
            }
            cur += 1;
        }
    }
    return this->settings[0];
}

int InteractiveSetting::Size(const SettingCollection &settings)
{
    int size = 0;
    for (const auto &setting : this->settings)
    {
        const auto &val = settings.Get(setting.first);
        if (val.IsVec2())
        {
            size += 2;
        }
        else if (val.IsVec3())
        {
            size += 3;
        }
        else
        {
            size += 1;
        }
    }
    return size;
}

void InteractiveSetting::Input(SettingCollection &settings, SDL_Event event)
{
    UiSetting::Input(settings, event);
    if (event.type == SDL_KEYDOWN)
    {
        auto code = event.key.keysym.scancode;
        if (code == down)
        {
            currentIndex++;
            if (currentIndex >= Size(settings))
            {
                currentIndex = 0;
            }
        }
        else if (code == up)
        {
            currentIndex--;
            if (currentIndex < 0)
            {
                currentIndex = Size(settings) - 1;
            }
        }
        else if (code == increase || code == decrease)
        {
            auto &item = settings.Get(Current(settings, nullptr).first);
            if (item.IsBool())
            {
                auto &value = item.AsBool();
                value = !value;
            }
        }
    }
}

void InteractiveSetting::Integrate(SettingCollection &settings, double time)
{
    UiSetting::Integrate(settings, time);
    int delta = 0;
    if (IsKey(increase))
    {
        delta++;
    }
    if (IsKey(decrease))
    {
        delta--;
    }
    if (delta != 0)
    {
        int elem = 0;
        auto &current = Current(settings, &elem);
        auto &item = settings.Get(current.first);
        if (item.IsInt())
        {
            auto &value = item.AsInt();
            value += delta;
        }
        else if (item.IsFloat() || item.IsVec2() || item.IsVec3())
        {
            double *value;
            if (item.IsFloat())
            {
                value = &item.AsFloat();
            }
            else if (item.IsVec2())
            {
                value = elem == 0 ? &item.AsVec2().x : &item.AsVec2().y;
            }
            else
            {
                value =
                    elem == 0 ? &item.AsVec3().x : elem == 1 ? &item.AsVec3().y
                                                             : &item.AsVec3().z;
            }
            auto rate = current.second;
            if (rate >= 0)
            {
                *value += delta * rate * time;
            }
            else
            {
                rate = -rate;
                *value *= exp(delta * rate * time);
            }
        }
    }
}

std::string
InteractiveSetting::Describe(const SettingCollection &settings) const
{
    std::ostringstream out;
    int idx = 0;
    for (const auto &setting : this->settings)
    {
        const auto &val = settings.Get(setting.first);
        int size = 1;
        if (val.IsVec2())
        {
            size = 2;
        }
        else if (val.IsVec3())
        {
            size = 3;
        }
        for (int delta = 0; delta < size; delta++)
        {
            if (idx == currentIndex)
            {
                out << "* ";
            }
            else
            {
                out << "  ";
            }
            out << val.Name();
            if (size != 1)
            {
                out << "[" << delta << "]";
            }
            out << " = ";
            if (val.IsInt())
            {
                out << val.AsInt();
            }
            else if (val.IsFloat())
            {
                out << val.AsFloat();
            }
            else if (val.IsBool())
            {
                out << val.AsBool();
            }
            else if (val.IsVec2())
            {
                out << (delta == 0 ? val.AsVec2().x : val.AsVec2().y);
            }
            else if (val.IsVec3())
            {
                out << (delta == 0 ? val.AsVec3().x : delta == 1 ? val.AsVec3()
                    .y : val.AsVec3().z);
            }
            out << std::endl;
            idx++;
        }
    }
    return out.str();
}
