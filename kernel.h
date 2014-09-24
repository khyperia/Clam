#pragma once

#include "context.h"
#include <map>

class ClamKernel
{
	std::shared_ptr<cl_command_queue> queue;
	std::shared_ptr<cl_kernel> kernel;
	size_t launchSize[2];
	
    template<int index>
	void RunImpl()
	{
		Invoke();
	}

	template<int index, typename ArgFirst, typename... ArgRest>
	void RunImpl(ArgFirst first, ArgRest... rest)
	{
		if (clSetKernelArg(*kernel, index, sizeof(ArgFirst), &first))
			throw std::runtime_error("Could not set kernel argument");
		RunImpl<index + 1, ArgRest...>(rest...);
	}

public:
    static std::map<std::string, std::shared_ptr<ClamKernel>> namedKernels;

	ClamKernel();
	ClamKernel(std::shared_ptr<cl_context> context, std::shared_ptr<cl_device_id> device, const char* sourcecode);
	void SetLaunchSize(size_t width, size_t height);

	std::shared_ptr<cl_command_queue> GetQueue()
	{
		return queue;
	}

	template<typename... Args>
	void Run(Args... args)
	{
		RunImpl<0>(args...);
	}

    // does NOT set arguments
	void Invoke();
    void SetArg(int index, int size, const void* data)
    {
		if (clSetKernelArg(*kernel, index, size, data))
			throw std::runtime_error("Could not set kernel argument " + std::to_string(index));
    }
};
