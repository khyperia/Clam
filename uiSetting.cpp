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

Camera3d::Camera3d(std::string pos_x,
                   std::string pos_y,
                   std::string pos_z,
                   std::string look_x,
                   std::string look_y,
                   std::string look_z,
                   std::string up_x,
                   std::string up_y,
                   std::string up_z,
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
    : pos_x(pos_x), pos_y(pos_y), pos_z(pos_z), look_x(look_x), look_y(look_y),
      look_z(look_z), up_x(up_x), up_y(up_y), up_z(up_z), fov(fov),
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
    auto &pos_x = settings.Get(this->pos_x).AsFloat();
    auto &pos_y = settings.Get(this->pos_y).AsFloat();
    auto &pos_z = settings.Get(this->pos_z).AsFloat();
    auto &look_x = settings.Get(this->look_x).AsFloat();
    auto &look_y = settings.Get(this->look_y).AsFloat();
    auto &look_z = settings.Get(this->look_z).AsFloat();
    auto &up_x = settings.Get(this->up_x).AsFloat();
    auto &up_y = settings.Get(this->up_y).AsFloat();
    auto &up_z = settings.Get(this->up_z).AsFloat();
    const auto move_speed = settings.Get(this->move_speed).AsFloat() * time;
    const auto turn_speed = settings.Get(this->fov).AsFloat() * time;
    auto pos = Vector3<double>(pos_x, pos_y, pos_z);
    auto look = Vector3<double>(look_x, look_y, look_z);
    auto up = Vector3<double>(up_x, up_y, up_z);
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
    pos_x = pos.x;
    pos_y = pos.y;
    pos_z = pos.z;
    look_x = look.x;
    look_y = look.y;
    look_z = look.z;
    up_x = up.x;
    up_y = up.y;
    up_z = up.z;
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

void InteractiveSetting::Input(SettingCollection &settings, SDL_Event event)
{
    UiSetting::Input(settings, event);
    if (event.type == SDL_KEYDOWN)
    {
        auto code = event.key.keysym.scancode;
        if (code == up)
        {
            currentIndex++;
            if (currentIndex >= (int)this->settings.size())
            {
                currentIndex = 0;
            }
        }
        else if (code == down)
        {
            currentIndex--;
            if (currentIndex < 0)
            {
                currentIndex = (int)this->settings.size();
            }
        }
        else if (code == increase || code == decrease)
        {
            auto &item = settings.Get(this->settings[currentIndex].first);
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
        auto &item = settings.Get(this->settings[currentIndex].first);
        if (item.IsInt())
        {
            auto &value = item.AsInt();
            value += delta;
        }
        else if (item.IsFloat())
        {
            auto &value = item.AsFloat();
            auto rate = this->settings[currentIndex].second;
            if (rate >= 0)
            {
                value += delta * rate * time;
            }
            else
            {
                rate = -rate;
                value *= exp(delta * rate * time);
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
        if (idx == currentIndex)
        {
            out << "* ";
        }
        else
        {
            out << "  ";
        }
        out << val.Name() << " = ";
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
        out << std::endl;
        idx++;
    }
    return out.str();
}
