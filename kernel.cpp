#include "kernel.h"
#include "helper.h"
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>

ClamKernel::ClamKernel()
{
}

ClamKernel::ClamKernel(std::shared_ptr<cl_context> context,
        std::shared_ptr<cl_device_id> device, const char* sourcecode)
{
	cl_int openclError = 0;
	cl_program openclProgram = clCreateProgramWithSource(*context, 1, &sourcecode, 0, &openclError);
	if (openclError)
		throw std::runtime_error("Failed to create program");

	openclError = clBuildProgram(openclProgram, 1, device.get(), "-cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -Werror", 0, 0);
	if (openclError == CL_BUILD_PROGRAM_FAILURE)
	{
		// Determine the size of the log
		size_t lLogSize;
		clGetProgramBuildInfo(
			openclProgram, *device, CL_PROGRAM_BUILD_LOG, 0, 0, &lLogSize
			);

		// Get the log
		std::string lLog;
		lLog.resize(lLogSize);
		clGetProgramBuildInfo(openclProgram, *device, CL_PROGRAM_BUILD_LOG, lLogSize, const_cast<char*>(lLog.data()), 0);

		puts("Kernel failed to compile.");
		puts(lLog.c_str());
	}
	if (openclError)
		throw std::runtime_error("Failed to build program");

    std::vector<cl_kernel> ckernels(16);
    unsigned int numKernels;
    clCreateKernelsInProgram(openclProgram, ckernels.size(), ckernels.data(), &numKernels);
    
    for (unsigned int i = 0; i < numKernels; i++)
    {
        std::vector<char> kernelNameVec(100);
        size_t kernelNameVecSize;
        clGetKernelInfo(ckernels[i], CL_KERNEL_FUNCTION_NAME,
                kernelNameVec.size(), kernelNameVec.data(), &kernelNameVecSize);
        std::string name(kernelNameVec.data(), kernelNameVecSize - 1);
        kernels[name] = std::make_shared<cl_kernel>(ckernels[i]);
    }

	cl_command_queue openclCommandQueue = clCreateCommandQueue(*context, *device, 0, &openclError);
	if (openclError)
		throw std::runtime_error("Failed to create command queue");

	queue = std::make_shared<cl_command_queue>(openclCommandQueue);

	SetLaunchSize(0, 0);
}

void ClamKernel::Invoke(std::string kernName)
{
    if (kernels.find(kernName) == kernels.end())
        throw std::runtime_error("Kernel did not exist: " + kernName);
	size_t local[] = { 8, 8 };
	size_t global[] = {
		(launchSize[0] + local[0] - 1) / local[0] * local[0],
		(launchSize[1] + local[1] - 1) / local[1] * local[1]
	};
	cl_int err = clEnqueueNDRangeKernel(*queue, *kernels[kernName],
            2, nullptr, global, local, 0, nullptr, nullptr);
	if (err)
		throw std::runtime_error("Failed to launch kernel");
}

void ClamKernel::SetLaunchSize(size_t width, size_t height)
{
	launchSize[0] = width;
	launchSize[1] = height;
}
