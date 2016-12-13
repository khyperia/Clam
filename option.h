#pragma once

#include <string>
#include <vector>

class UserOptions
{
    std::vector<std::pair<std::string, std::string>> values;
public:
    void Add(std::string name, std::string default_value);
    std::string Parse(int argc, char **argv);
    const std::string &Get(std::string name);
};
