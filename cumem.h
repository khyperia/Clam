#pragma once

#include "util.h"
#include "cudaContext.h"
#include <vector>
#include <iostream>

template<typename T>
class CuMem : public NoClone
{
    CUdeviceptr ptr;
    size_t count;
    bool owned;

    void Alloc(const CudaContext &context)
    {
        context.Run(cuMemAlloc(&ptr, bytesize()));
    }

    void Free()
    {
        if (ptr)
        {
            if (cuMemFree(ptr) != CUDA_SUCCESS)
            {
                std::cout << "Could not free CuMem" << std::endl;
            }
            ptr = 0;
        }
    }

public:
    CuMem() : ptr(0), count(0), owned(false)
    {
    }

    CuMem(const CudaContext &context, size_t count) : count(count), owned(true)
    {
        Alloc(context);
    }

    CuMem(CUdeviceptr ptr, size_t count) : ptr(ptr), count(count), owned(false)
    {
    }

    CuMem(const CuMem &) = delete;
    CuMem(CuMem &&) = default;

    ~CuMem()
    {
        if (owned)
        {
            Free();
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

    void CopyTo(T *cpu, CUstream stream, const CudaContext &context) const
    {
        if (!ptr)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        context.Run(cuMemcpyDtoHAsync(cpu, ptr, bytesize(), stream));
    }

    void CopyFrom(const T *cpu, CUstream stream, const CudaContext &context) const
    {
        if (!ptr)
        {
            throw std::runtime_error("Operating on invalid CuMem");
        }
        context.Run(cuMemcpyHtoDAsync(ptr, cpu, bytesize(), stream));
    }
};
