#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <memory>

class ClamContext
{
    std::shared_ptr<cl_context> context;
    std::shared_ptr<cl_device_id> device;
    public:
    ClamContext();
    std::shared_ptr<cl_context> GetContext()
    {
        return context;
    }
    std::shared_ptr<cl_device_id> GetDevice()
    {
        return device;
    }
};
