#pragma once

#include <sstream>
#include <CL/cl.h>

#define HandleCl(x) HandleClImpl(x, #x, __FILE__, __LINE__)

static inline void HandleClImpl(cl_int err, const char* expr, const char* file, const int line)
{
    if (err != CL_SUCCESS)
    {
        std::ostringstream msg;
        msg << "OpenCL Error: " << err << std::endl;
        msg << file << "(" << line << "): " << expr;
        throw std::runtime_error(msg.str());
    }
}

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
