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

    CuMem(const CuMem &)
    {
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

    void Realloc(size_t newCount)
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

    void CopyTo(T *cpu, CUstream stream) const
    {
        if (!ptr)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        HandleCu(cuMemcpyDtoHAsync(cpu, ptr, bytesize(), stream));
    }

    void CopyFrom(const T *cpu, CUstream stream) const
    {
        if (!ptr)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        HandleCu(cuMemcpyHtoDAsync(ptr, cpu, bytesize(), stream));
    }

    std::vector<T> Download(CUstream stream) const
    {
        std::vector<T> result(elemsize());
        CopyTo(result.data(), stream);
        cuStreamSynchronize(stream);
        return result;
    }

    static CuMem<T> Upload(const std::vector<T> &cpu, CUstream stream)
    {
        CuMem<T> result(cpu.size());
        result.CopyFrom(cpu.data(), stream);
        cuStreamSynchronize(stream);
        return result;
    }
};
