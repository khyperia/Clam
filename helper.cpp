#include "helper.h"
#include <stdexcept>

void HandleErrImpl(int errcode, const char* filename, int line)
{
    if (errcode != 0)
        throw std::runtime_error(std::string(filename));
}
