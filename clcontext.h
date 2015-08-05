#pragma once

#include <CL/cl.h>
#include <string>

cl_context GetDevice(std::string platformName, bool wireToDisplay);
