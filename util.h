#pragma once

#include <cuda.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

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

// not allowed to clone nor move
class Immobile
{
protected:
    Immobile() = default;
    virtual ~Immobile() = default;

    Immobile(Immobile const &) = delete;
    Immobile(Immobile &&) = delete;
    Immobile& operator=(Immobile const &x) = delete;
    Immobile& operator=(Immobile &&x) = delete;
};

// allowed to move, but not clone
class NoClone
{
protected:
    NoClone() = default;
    virtual ~NoClone() = default;
    NoClone(NoClone &&) = default;
    NoClone& operator=(NoClone &&x) = default;

    NoClone(NoClone const &) = delete;
    NoClone& operator=(NoClone const &x) = delete;
};

template<typename T, typename Tscalar>
T CatmullRom(T p0, T p1, T p2, T p3, Tscalar t)
{
    Tscalar t2 = t * t;
    Tscalar t3 = t2 * t;

    return (((Tscalar)2 * p1) + (-p0 + p2) * t
        + ((Tscalar)2 * p0 - (Tscalar)5 * p1 + (Tscalar)4 * p2 - p3) * t2
        + (-p0 + (Tscalar)3 * p1 - (Tscalar)3 * p2 + p3) * t3) / (Tscalar)2;
}

class CudaContext: public NoClone
{
    int deviceIndex;
    CUdevice device;
    CUcontext context;
    static int currentContext;

    inline static void CheckCall(CUresult callResult)
    {
        if (callResult != CUDA_SUCCESS)
        {
            const char *errstr;
            if (cuGetErrorString(callResult, &errstr) == CUDA_SUCCESS)
            {
                throw std::runtime_error("CUDA error (" + tostring(callResult) + "): " + errstr);
            }
            else if (cuGetErrorName(callResult, &errstr) == CUDA_SUCCESS)
            {
                throw std::runtime_error("CUDA error " + tostring(callResult) + " = " + errstr);
            }
            else
            {
                throw std::runtime_error("CUDA error " + tostring(callResult) + " (no name)");
            }
        }
    }

public:
    static void Init();
    static inline void RunWithoutContext(CUresult callResult)
    {
        CheckCall(callResult);
    }

    static int DeviceCount();

    CudaContext(int deviceIndex);

    CUcontext Context() const;
    CUdevice Device() const;
    std::string DeviceName() const;
    void SetCurrent() const;

    inline void Run(CUresult callResult) const
    {
        if (currentContext != deviceIndex)
        {
            throw std::runtime_error(
                "Current context not correct, call CudaContext::SetCurrent() before this call. This context: "
                    + tostring(deviceIndex) + ", active context: "
                    + tostring(currentContext));
        }
        CheckCall(callResult);
    }
};
