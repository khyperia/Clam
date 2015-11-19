#pragma once

#include <sstream>
#include <stdexcept>
#include <cuda.h>
#include <iostream>

#define HandleCu(x) HandleCuImpl(x, #x, __FILE__, __LINE__)

static inline void HandleCuImpl(CUresult err, const char* expr, const char* file, const int line)
{
    if (err != CUDA_SUCCESS)
    {
        std::ostringstream msg;
        const char* errstr;
        if (cuGetErrorString(err, &errstr) != CUDA_SUCCESS)
        {
            errstr = "Unknown error";
        }
        msg << "CUDA Error (" << err << "): " << errstr << "\n" << file << "(" << line << "): " << expr;
        throw std::runtime_error(msg.str());
    }
}

template<typename T, typename Tscalar>
T CatmullRom(T p0, T p1, T p2, T p3, Tscalar t)
{
    Tscalar t2 = t * t;
    Tscalar t3 = t2 * t;

    return (((Tscalar)2 * p1) +
            (-p0 + p2) * t +
            ((Tscalar)2 * p0 - (Tscalar)5 * p1 + (Tscalar)4 * p2 - p3) * t2 +
            (-p0 + (Tscalar)3 * p1 - (Tscalar)3 * p2 + p3) * t3) / (Tscalar)2;
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
