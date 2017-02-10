#include "option.h"
#include <algorithm>
#include <sstream>

void UserOptions::Add(std::string name, std::string default_value)
{
    values.push_back(std::make_pair(name, default_value));
}

std::string UserOptions::Parse(int, char **argv)
{
    for (auto &value : values)
    {
        auto upper = value.first;
        std::transform(upper.begin(), upper.end(), upper.begin(), (int (*)(int))std::toupper);
        const auto env_name = "CLAM3_" + upper;
        auto env_val = std::getenv(env_name.c_str());
        if (env_val)
        {
            value.second = std::string(env_val);
        }
    }
    argv++;
    while (*argv)
    {
        std::string arg = *argv;
        bool had_prefix = false;
        while (arg.size() > 0 && (arg[0] == '-' || arg[0] == '/'))
        {
            arg = arg.substr(1);
            had_prefix = true;
        }
        if (!had_prefix)
        {
            return "Option " + arg + " missing argument prefix";
        }
        if (arg == "help")
        {
            std::ostringstream help;
            help << "Clam3 help:" << std::endl;
            for (auto &value : values)
            {
                help << "--";
                help << value.first;
                help << " ";
                help << value.second;
                help << std::endl;
            }
            return help.str();
        }
        argv++;
        if (!*argv)
        {
            return "Option " + arg + " missing value";
        }
        auto arg_value = *argv;
        argv++;
        bool found = false;
        for (auto &value : values)
        {
            if (arg == value.first)
            {
                value.second = arg_value;
                found = true;
            }
        }
        if (!found)
        {
            return "Option \"" + arg + "\" does not exist";
        }
    }
    return "";
}

const std::string &UserOptions::Get(std::string name)
{
    for (const auto &value : values)
    {
        if (value.first == name)
        {
            return value.second;
        }
    }
    static std::string empty("");
    return empty;
}
