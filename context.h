#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include "helper.h"

class ClamContext
{
    std::shared_ptr<cl::Context> context;
    std::shared_ptr<cl::Device> device;
    public:
    ClamContext();
    std::shared_ptr<cl::Context> GetContext()
    {
        return context;
    }
    std::shared_ptr<cl::Device> GetDevice()
    {
        return device;
    }
};
