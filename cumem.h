#pragma once

#include "util.h"
#include <vector>
#include <iostream>

template<typename T>
class CuMem
{
    CUdeviceptr ptr;
    size_t count;
    bool owned;

    CuMem &operator=(const CuMem &)
    {
        return *this;
    }

    void Alloc()
    {
        HandleCu(cuMemAlloc(&ptr, bytesize()));
    }

    void Free()
    {
        if (ptr)
        {
            if (cuMemFree(ptr) != CUDA_SUCCESS)
            {
                std::cout << "Could not free CuMem\n";
            }
            ptr = 0;
        }
    }

public:
    // oh god I hope this doesn't get used, it's only public becase std::vector needs it to fill it
    CuMem(const CuMem &other)
    {
        ptr = other.ptr;
        count = other.count;
        owned = other.owned;
    }

    CuMem() : ptr(0), count(0), owned(false)
    {
    }

    CuMem(size_t count) : count(count), owned(true)
    {
        Alloc();
    }

    CuMem(CUdeviceptr ptr, size_t count) : ptr(ptr), count(count), owned(false)
    {
    }

    ~CuMem()
    {
        if (owned)
        {
            Free();
        }
    }

    void Realloc(size_t newCount, int context)
    {
        (void)context;
        if (ptr != 0)
        {
            if (!owned)
            {
                throw std::runtime_error("Cannot resize a non-owned CuMem");
            }
            Free();
        }
        else
        {
            owned = true;
        }
        count = newCount;
        Alloc();
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

    void CopyTo(T *cpu, CUstream stream, int context) const
    {
        (void)context;
        if (!ptr)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        HandleCu(cuMemcpyDtoHAsync(cpu, ptr, bytesize(), stream));
    }

    void CopyFrom(const T *cpu, CUstream stream, int context) const
    {
        (void)context;
        if (!ptr)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        HandleCu(cuMemcpyHtoDAsync(ptr, cpu, bytesize(), stream));
    }
};
