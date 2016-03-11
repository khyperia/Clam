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

    void Alloc(CudaContext context)
    {
        context.Run(cuMemAlloc(&ptr, bytesize()));
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

    CuMem(CudaContext context, size_t count) : count(count), owned(true)
    {
        Alloc(context);
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

    void Realloc(size_t newCount, CudaContext context)
    {
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
        Alloc(context);
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

    void CopyTo(T *cpu, CUstream stream, CudaContext context) const
    {
        if (!ptr)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        context.Run(cuMemcpyDtoHAsync(cpu, ptr, bytesize(), stream));
    }

    void CopyFrom(const T *cpu, CUstream stream, CudaContext context) const
    {
        if (!ptr)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        context.Run(cuMemcpyHtoDAsync(ptr, cpu, bytesize(), stream));
    }
};
