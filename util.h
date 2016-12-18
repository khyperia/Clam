#pragma once

#include "movable.h"
#include <memory>
#include <sstream>

class TimeEstimate: public Immobile
{
    struct Time;
    std::unique_ptr<Time> start;
public:
    TimeEstimate();
    ~TimeEstimate();
    std::string Mark(int current, int total);
};

template<typename T>
static std::string tostring(const T &value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

template<typename T>
static T fromstring(const std::string &value)
{
    std::istringstream stream(value);
    T result;
    stream >> result;
    return result;
}

template<typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args &&...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
