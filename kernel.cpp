#include "kernel.h"
#include <stdexcept>
#include <fstream>
#include <string>

using namespace cl;

ClamKernel::ClamKernel()
{
}

ClamKernel::ClamKernel(std::shared_ptr<cl::Context> context, std::shared_ptr<cl::Device> device, const char* filename)
{
    std::ifstream file(filename);
    std::string filecontents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    Program::Sources source;
    source.push_back(std::make_pair(filecontents.c_str(), filecontents.length()));

    Program program(*context, source);

    std::vector<Device> devices;
    devices.push_back(*device);
    try
    {
        program.build(devices, "-cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -Werror");
    }
    catch (const cl::Error& err)
    {
        printf("%s\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]).c_str());
        throw;
    }
    
    puts("Kernel build log:");
    puts(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]).c_str());

    cl_int err = CL_SUCCESS;

    kernel = std::make_shared<Kernel>(program, "main", &err);

    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("Failed to create kernel");
    }

    queue = std::make_shared<CommandQueue>(*context, *device, 0, &err);

    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("Failed to create CommandQueue");
    }

    launchSize = NullRange;
}

void ClamKernel::RunImpl(int numArgs)
{
    cl::NDRange local(4, 4);
    cl::NDRange global(
            (launchSize[0] + local[0] - 1) / local[0] * local[0],
            (launchSize[1] + local[1] - 1) / local[1] * local[1]);
    HandleErr(queue->enqueueNDRangeKernel(*kernel, NullRange, global, local, nullptr, nullptr));
}

void ClamKernel::SetLaunchSize(NDRange range)
{
    launchSize = range;
}
