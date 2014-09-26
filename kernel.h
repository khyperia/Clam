#pragma once

#include "context.h"
#include <map>

class ClamKernel
{
	std::shared_ptr<cl_command_queue> queue;
    std::map<std::string, std::shared_ptr<cl_kernel>> kernels;
	size_t launchSize[2];
public:
	ClamKernel();
	ClamKernel(std::shared_ptr<cl_context> context,
            std::shared_ptr<cl_device_id> device, const char* sourcecode);
	void SetLaunchSize(size_t width, size_t height);

	std::shared_ptr<cl_command_queue> GetQueue()
	{
		return queue;
	}

    // Invoke does NOT set arguments
	void Invoke(std::string kernName);
    void SetArg(std::string kernName, int index, int size, const void* data)
    {
        if (kernels.find(kernName) == kernels.end())
            throw std::runtime_error("Kernel did not exist: " + kernName);
		if (int openclError = clSetKernelArg(*kernels[kernName], index, size, data))
			throw std::runtime_error("Could not set kernel argument " + std::to_string(index) + ": " + std::to_string(openclError));
    }
};
