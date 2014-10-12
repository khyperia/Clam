#pragma once

#include "context.h"
#include <map>
#include <vector>

class ClamKernel
{
    std::shared_ptr<cl_command_queue> queue;
    std::map<std::string, std::shared_ptr<cl_kernel>> kernels;
public:
    ClamKernel();
    ClamKernel(std::shared_ptr<cl_context> context, std::shared_ptr<cl_device_id> device,
            std::shared_ptr<cl_command_queue> queue, std::vector<std::string> const& sourcecode);

    // Invoke does NOT set arguments
    void Invoke(std::string kernName, unsigned long launchWidth, unsigned long launchHeight);
    void SetArg(std::string kernName, cl_uint index, size_t size, const void* data)
    {
        if (kernels.find(kernName) == kernels.end())
            throw std::runtime_error("Kernel did not exist: " + kernName);
        if (int openclError = clSetKernelArg(*kernels[kernName], index, size, data))
            throw std::runtime_error("Could not set kernel argument " + std::to_string(index) + ": " + std::to_string(openclError));
    }
};
