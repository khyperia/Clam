#pragma once

#include "util.h"
#include <vector>
#include <iostream>

template<typename T>
class CuMem
{
    CUdeviceptr ptr;
    int *refcount;
    size_t count;
public:
    CuMem() : ptr(0), refcount(NULL), count(0)
    {
    }

    CuMem(size_t count) : refcount(new int(1)), count(count)
    {
        HandleCu(cuMemAlloc(&ptr, count * sizeof(T)));
    }

    CuMem(const CuMem &x) :
            ptr(x.ptr),
            refcount(x.refcount),
            count(x.count)
    {
        if (refcount)
        {
            ++*refcount;
        }
    }

    CuMem& operator=(const CuMem& rhs)
    {
        ptr = rhs.ptr;
        refcount = rhs.refcount;
        count = rhs.count;
        if (refcount)
        {
            ++*refcount;
        }
        return *this;
    }

    ~CuMem()
    {
        if (refcount && !--*refcount)
        {
            delete refcount;
            HandleCu(cuMemFree(ptr));
        }
    }

    CUdeviceptr &operator()()
    {
        return ptr;
    }

    size_t elemsize() const
    {
        return count;
    }

    size_t bytesize() const
    {
        return elemsize() * sizeof(T);
    }

    void CopyTo(T *cpu) const
    {
        if (!refcount)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        HandleCu(cuMemcpyDtoH(cpu, ptr, bytesize()));
    }

    void CopyFrom(const T *cpu) const
    {
        if (!refcount)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        HandleCu(cuMemcpyHtoD(ptr, cpu, bytesize()));
    }

    std::vector<T> Download() const
    {
        std::vector<T> result(elemsize());
        CopyTo(result.data());
        return result;
    }

    static CuMem<T> Upload(const std::vector<T> &cpu)
    {
        CuMem<T> result(cpu.size());
        result.CopyFrom(cpu.data());
        return result;
    }
};