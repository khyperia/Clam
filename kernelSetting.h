#pragma once

#include <vector>
#include "util.h"

class Setting: public NoClone
{
    const std::string name;
    enum TagTy
    {
        FLOAT, INT, BOOL
    } tag;
    union
    {
        double value_float;
        int value_int;
        bool value_bool;
    };
    bool IsType(TagTy type) const;
    void Check(TagTy type) const;

public:
    Setting(const std::string name, double value);
    Setting(const std::string name, int value);
    Setting(const std::string name, bool value);
    ~Setting();
    Setting(Setting &&) = default;
    const std::string &Name() const;
    int &AsInt();
    double &AsFloat();
    bool &AsBool();
    const int &AsInt() const;
    const double &AsFloat() const;
    const bool &AsBool() const;

    std::string Save() const;
    void Load(const std::string &savedData);

    static Setting Interpolate(const Setting &p0,
                        const Setting &p1,
                        const Setting &p2,
                        const Setting &p3,
                        double time);
};

class SettingCollection: public NoClone
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
    const std::string Save();
    void Load(const std::string &savedData);
    Setting &Get(const std::string &name);
    const Setting &Get(const std::string &name) const;
    static SettingCollection Interpolate(SettingCollection p0,
                                         SettingCollection p1,
                                         SettingCollection p2,
                                         SettingCollection p3,
                                         double time);
};
