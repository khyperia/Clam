#pragma once

#include <sstream>
#include <stdexcept>
#include <cuda.h>
#include <iostream>

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

class CudaContext
{
    CUcontext context;
    int index;
    static int currentContext;

public:
    static const CudaContext Invalid;

    CudaContext(CUcontext context, int index) : context(context), index(index)
    {
    }

    CUcontext UnderlyingContext() const
    {
        return context;
    }

    int Index() const
    {
        if (IsValid())
        {
            return index;
        }
        throw std::runtime_error("Index() called on invalid context");
    }

    inline void Run(CUresult callResult) const
    {
        if (!IsValid())
        {
            throw std::runtime_error("Run() called on invalid context");
        }
        if (currentContext != index)
        {
            throw std::runtime_error(
                    "Current context not correct, call CudaContext::SetCurrent() before this call. This context: "
                    + tostring(index) + ", active context: " + tostring(currentContext));
        }
        if (callResult != CUDA_SUCCESS)
        {
            const char *errstr;
            //if (cuGetErrorString(callResult, &errstr) != CUDA_SUCCESS)
            {
                errstr = "unknown error";
            }
            throw std::runtime_error("CUDA error (" + tostring(callResult) + "): " + errstr);
        }
    }

    void SetCurrent() const
    {
        if (!IsValid())
        {
            throw std::runtime_error("SetCurrent() called on invalid context");
        }
        if (currentContext == index)
        {
            return;
        }
        currentContext = index;
        Run(cuCtxSetCurrent(context));
        //std::cout << "Set current context to " << index << std::endl;
    }

    bool IsValid() const
    {
        return index != -1;
    }
};

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
