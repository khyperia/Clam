#pragma once

#include "util.h"
#include <vector>
#include <iostream>

template<typename T>
class CuMem
{
    CUdeviceptr ptr;
    size_t count;

    CuMem &operator=(const CuMem &rhs)
    {
        return *this;
    }

    CuMem(const CuMem &x)
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
    CuMem() : ptr(0), count(0)
    {
    }

    CuMem(size_t count) : count(count)
    {
        Alloc();
    }

    CuMem(CUdeviceptr ptr, size_t count) : ptr(ptr), count(count)
    {
    }

    ~CuMem()
    {
        Free();
    }

    void Realloc(size_t newCount)
    {
        Free();
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

    void CopyTo(T *cpu) const
    {
        if (!ptr)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        HandleCu(cuMemcpyDtoH(cpu, ptr, bytesize()));
    }

    void CopyFrom(const T *cpu) const
    {
        if (!ptr)
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
