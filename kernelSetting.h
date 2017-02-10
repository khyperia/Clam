#pragma once

#include <vector>
#include "util.h"
#include "vector.h"

class Setting : public NoClone
{
    const std::string name;
    enum TagTy
    {
        FLOAT, INT, BOOL, VEC2, VEC3
    } tag;
    union
    {
        double value_float;
        int value_int;
        bool value_bool;
        Vector2<double> value_vec2;
        Vector3<double> value_vec3;
    };
    bool IsType(TagTy type) const;
    void Check(TagTy type) const;
public:
    Setting(const std::string name, double value);
    Setting(const std::string name, int value);
    Setting(const std::string name, bool value);
    Setting(const std::string name, Vector2<double> value);
    Setting(const std::string name, Vector3<double> value);
    ~Setting();
    Setting(Setting &&) = default;
    Setting Clone() const;
    const std::string &Name() const;
    bool IsInt() const;
    bool IsFloat() const;
    bool IsBool() const;
    bool IsVec2() const;
    bool IsVec3() const;
    int &AsInt();
    double &AsFloat();
    bool &AsBool();
    Vector2<double> &AsVec2();
    Vector3<double> &AsVec3();
    const int &AsInt() const;
    const double &AsFloat() const;
    const bool &AsBool() const;
    const Vector2<double> &AsVec2() const;
    const Vector3<double> &AsVec3() const;
    std::string Save() const;
    void Load(const std::string &savedData);
    static Setting Interpolate(
        const Setting &p0, const Setting &p1, const Setting &p2, const Setting &p3, double time
    );
};

class SettingCollection : public NoClone
{
    std::vector<Setting> settings;
    void AddSetting(Setting setting);
public:
    SettingCollection();
    ~SettingCollection();
    SettingCollection(SettingCollection &&) = default;
    void AddSetting(const std::string name, double value);
    void AddSetting(const std::string name, int value);
    void AddSetting(const std::string name, bool value);
    void AddSetting(const std::string name, Vector2<double> value);
    void AddSetting(const std::string name, Vector3<double> value);
    SettingCollection Clone() const;
    const std::string Save();
    void Load(const std::string &savedData);
    Setting &Get(const std::string &name);
    const Setting &Get(const std::string &name) const;
    static SettingCollection Interpolate(
        const SettingCollection &p0,
        const SettingCollection &p1,
        const SettingCollection &p2,
        const SettingCollection &p3,
        double time
    );
};
