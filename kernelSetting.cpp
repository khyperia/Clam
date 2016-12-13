#include "kernelSetting.h"

template<typename T, typename Tscalar>
T CatmullRom(T p0, T p1, T p2, T p3, Tscalar t)
{
    Tscalar t2 = t * t;
    Tscalar t3 = t2 * t;

    return (((Tscalar)2 * p1) + (-p0 + p2) * t
        + ((Tscalar)2 * p0 - (Tscalar)5 * p1 + (Tscalar)4 * p2 - p3) * t2
        + (-p0 + (Tscalar)3 * p1 - (Tscalar)3 * p2 + p3) * t3) / (Tscalar)2;
}

Setting::Setting(const std::string name, double value)
    : name(name), tag(FLOAT), value_float(value)
{
}

Setting::Setting(const std::string name, int value)
    : name(name), tag(INT), value_int(value)
{
}

Setting::Setting(const std::string name, bool value)
    : name(name), tag(BOOL), value_bool(value)
{
}

Setting::~Setting()
{
}

bool Setting::IsType(Setting::TagTy type) const
{
    return type == tag;
}

void Setting::Check(Setting::TagTy type) const
{
    if (!IsType(type))
    {
        throw std::runtime_error(
            "Tried to access " + name + " (" + tostring(tag) + ") as a "
                + tostring(type));
    }
}

const std::string &Setting::Name() const
{
    return name;
}

bool Setting::IsInt() const
{
    return IsType(INT);
}

bool Setting::IsFloat() const
{
    return IsType(FLOAT);
}

bool Setting::IsBool() const
{
    return IsType(BOOL);
}

int &Setting::AsInt()
{
    Check(INT);
    return value_int;
}

double &Setting::AsFloat()
{
    Check(FLOAT);
    return value_float;
}

bool &Setting::AsBool()
{
    Check(BOOL);
    return value_bool;
}

const int &Setting::AsInt() const
{
    Check(INT);
    return value_int;
}

const double &Setting::AsFloat() const
{
    Check(FLOAT);
    return value_float;
}

const bool &Setting::AsBool() const
{
    Check(BOOL);
    return value_bool;
}

Setting Setting::Interpolate(const Setting &p0,
                             const Setting &p1,
                             const Setting &p2,
                             const Setting &p3,
                             double time)
{
    if (p0.name != p1.name || p0.tag != p1.tag || p0.name != p2.name
        || p0.tag != p2.tag || p0.name != p3.name || p0.tag != p3.tag)
    {
        throw std::runtime_error("Invalid interpolation");
    }
    switch (p0.tag)
    {
        case FLOAT:
            return Setting(p0.name,
                           CatmullRom(p0.AsFloat(),
                                      p1.AsFloat(),
                                      p2.AsFloat(),
                                      p3.AsFloat(),
                                      time));
        case INT:
            return Setting(p0.name,
                           CatmullRom(p0.AsInt(),
                                      p1.AsInt(),
                                      p2.AsInt(),
                                      p3.AsInt(),
                                      time));
        case BOOL:
            return Setting(p0.name, p1.AsBool());
    }
    throw std::runtime_error("Invalid tag for " + p0.name);
}

std::string Setting::Save() const
{
    switch (tag)
    {
        case FLOAT:
            return name + " " + tostring(AsFloat());
        case INT:
            return name + " " + tostring(AsInt());
        case BOOL:
            return name + (AsBool() ? " true" : " false");
    }
    throw std::runtime_error("Invalid tag for " + name);
}

void Setting::Load(const std::string &savedData)
{
    switch (tag)
    {
        case FLOAT:
            AsFloat() = fromstring<double>(savedData);
            break;
        case INT:
            AsInt() = fromstring<int>(savedData);
            break;
        case BOOL:
            AsBool() = (savedData == "true");
            break;
    }
}

SettingCollection::SettingCollection()
    : settings()
{
}

SettingCollection::~SettingCollection()
{
}

void SettingCollection::AddSetting(Setting setting)
{
    settings.push_back(std::move(setting));
}

void SettingCollection::AddSetting(const std::string name, double value)
{
    AddSetting(Setting(name, value));
}

void SettingCollection::AddSetting(const std::string name, int value)
{
    AddSetting(Setting(name, value));
}

void SettingCollection::AddSetting(const std::string name, bool value)
{
    AddSetting(Setting(name, value));
}

const std::string SettingCollection::Save()
{
    std::ostringstream out;
    for (const auto &value : settings)
    {
        out << value.Save() << std::endl;
    }
    return out.str();
}

void SettingCollection::Load(const std::string &savedData)
{
    std::istringstream in(savedData);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream linestream(line);
        std::string name;
        std::string value;
        std::getline(linestream, name, ' ');
        std::getline(linestream, value, ' ');
        if (name.empty() || value.empty())
        {
            throw std::runtime_error("Invalid formatted config line: " + line);
        }
        Get(name).Load(value);
    }
}

Setting &SettingCollection::Get(const std::string &name)
{
    for (auto &value : settings)
    {
        if (value.Name() == name)
        {
            return value;
        }
    }
    throw std::runtime_error("Could not find setting " + name);
}

const Setting &SettingCollection::Get(const std::string &name) const
{
    for (const auto &value : settings)
    {
        if (value.Name() == name)
        {
            return value;
        }
    }
    throw std::runtime_error("Could not find setting " + name);
}

SettingCollection SettingCollection::Interpolate(SettingCollection p0,
                                                 SettingCollection p1,
                                                 SettingCollection p2,
                                                 SettingCollection p3,
                                                 double time)
{
    SettingCollection result;
    for (const auto &v0 : p0.settings)
    {
        const auto &v1 = p1.Get(v0.Name());
        const auto &v2 = p2.Get(v1.Name());
        const auto &v3 = p3.Get(v2.Name());
        result.AddSetting(Setting::Interpolate(v0, v1, v2, v3, time));
    }
    return result;
}
