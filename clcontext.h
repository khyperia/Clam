#pragma once

#include <CL/cl.hpp>

cl::Context GetDevice(std::string platformName, bool wireToDisplay);
