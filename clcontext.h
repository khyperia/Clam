#pragma once

#include <CL/cl.hpp>

cl::Context GetDevice(std::string platname, std::string devname, bool wireToDisplay);
