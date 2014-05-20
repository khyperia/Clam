#pragma once

#include "context.h"

class ClamKernel
{
    std::shared_ptr<cl::CommandQueue> queue;
    std::shared_ptr<cl::Kernel> kernel;
    cl::NDRange launchSize;

    void RunImpl(int index);

    template<typename ArgFirst, typename... ArgRest>
        void RunImpl(int index, ArgFirst first, ArgRest... rest)
        {
            kernel->setArg(index++, first);
            RunImpl(index, rest...);
        }

    public:
    ClamKernel();
    ClamKernel(std::shared_ptr<cl::Context> context, std::shared_ptr<cl::Device> device, const char* filename);
    void SetLaunchSize(cl::NDRange range);

    std::shared_ptr<cl::CommandQueue> GetQueue()
    {
        return queue;
    }

    template<typename... Args>
        void Run(Args... args)
        {
            RunImpl(0, args...);
        }
};
